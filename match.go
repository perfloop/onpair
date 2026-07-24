package onpair

import (
	"bytes"
	"encoding/binary"
	"hash/maphash"
	"math"
	"math/bits"
	"unsafe"
)

// Bit masks for extracting prefixes of different lengths (little-endian)
var masks = [9]uint64{
	0x0000000000000000, // 0 bytes
	0x00000000000000FF, // 1 byte
	0x000000000000FFFF, // 2 bytes
	0x0000000000FFFFFF, // 3 bytes
	0x00000000FFFFFFFF, // 4 bytes
	0x000000FFFFFFFFFF, // 5 bytes
	0x0000FFFFFFFFFFFF, // 6 bytes
	0x00FFFFFFFFFFFFFF, // 7 bytes
	0xFFFFFFFFFFFFFFFF, // 8 bytes
}

const (
	minMatch              = 8
	maxOnPair16BucketSize = 128
)

// matcher is a hybrid longest prefix matcher supporting patterns up to
// 8+65535 bytes (insert rejects longer ones).
//
// Combines direct table lookup for short patterns with bucketed search for
// long patterns. Both paths are gated by 2-byte-prefix filters so find()
// skips the table probe entirely when no stored pattern starts with those
// bytes.
type matcher struct {
	longMatchBuckets longBucketTable // 8-byte prefix → candidate bucket (sorted desc by suffix length)
	shortMatchLookup [9]u64U16Table  // length → (prefix, token ID)
	lengthByPrefix2  *[65536]uint8   // bit L set ⇒ some short token of length L starts with these 2 LE bytes
	longBits2        *[1024]uint64   // bit set ⇒ some long token starts with these 2 LE bytes (65536-bit set)
	dictionary       []byte          // Suffix storage for long patterns
	endPositions     []uint32        // Boundary positions in dictionary
	onPair16         bool
	bucketSizeLimit  int
	groupSeed        uint64
}

// longBucket keeps small buckets ordered in place. Larger default buckets use
// an open-addressed key index. The key includes up to two suffix-head windows
// so a large group with one first head does not turn every lookup into a tail
// scan.
type longBucket struct {
	// The bounded pre-index form mirrors the pre-image's hot struct-of-arrays
	// scan. It is converted once at the fixed default-bucket threshold.
	heads      []uint64
	suffixLens []uint16
	ids        []uint16
	dictStarts []uint32

	candidates []longBucketCandidate // populated only after the threshold transition
	previous   []uint16              // prior candidate in the same keyed group, or noLongBucketCandidate
	groupSlots []uint16              // open-addressed group key → latest candidate slot; zero means empty
	groupCount int
	keyLenMask uint32 // bit L set => a group keyed by min(suffixLen, 16) == L exists
}

const (
	noLongBucketCandidate            = ^uint16(0)
	longBucketIndexThreshold         = maxOnPair16BucketSize
	initialLongBucketGroupSlotCount  = 2 * maxOnPair16BucketSize
	longBucketGroupLoadNumerator     = 7
	longBucketGroupLoadDenominator   = 8
	longBucketGroupGrowthNumerator   = 5
	longBucketGroupGrowthDenominator = 2
	longBucketGroupKeyBytes          = 2 * minMatch
)

type longBucketCandidate struct {
	// key is the packed first suffix head before indexing, then is replaced at
	// the fixed threshold with a process-keyed two-window group key.
	key       uint64
	dictStart uint32 // offset in m.dictionary where this suffix starts
	suffixLen uint16 // bytes past the 8-byte prefix
	id        uint16
}

func (b *longBucket) len() int {
	if b.candidates != nil {
		return len(b.candidates)
	}
	return len(b.heads)
}

func suffixKeyLen(suffixLen uint16) int {
	keyLen := int(suffixLen)
	if keyLen > longBucketGroupKeyBytes {
		return longBucketGroupKeyBytes
	}
	return keyLen
}

// newLongBucketGroupSeed is made once for each matcher, before the hot
// lookup path. A caller cannot carry an observed collision cluster from one
// trained model into another matcher instance.
func newLongBucketGroupSeed() uint64 {
	var hash maphash.Hash
	hash.SetSeed(maphash.MakeSeed())
	hash.WriteString("github.com/seiflotfy/onpair/longBucketGroup")
	return hash.Sum64()
}

func mixLongBucketKey(value uint64) uint64 {
	value ^= value >> 30
	value *= 0xbf58476d1ce4e5b9
	value ^= value >> 27
	value *= 0x94d049bb133111eb
	return value ^ (value >> 31)
}

// longBucketGroupKey uses the first two suffix-head windows. The full suffix
// is still checked before returning a match, so an exceptionally rare keyed
// hash collision can only share a chain, never change matching semantics.
func longBucketGroupKey(suffix []byte, seed uint64) uint64 {
	keyLen := len(suffix)
	if keyLen > longBucketGroupKeyBytes {
		keyLen = longBucketGroupKeyBytes
	}
	firstLen := keyLen
	if firstLen > minMatch {
		firstLen = minMatch
	}
	// Mix each window with the matcher seed before combining them. Applying the
	// seed only after an unkeyed XOR would leave an attacker able to manufacture
	// distinct first/second-window pairs with the same intermediate value.
	key := mixLongBucketKey(bytesToU64LE(suffix, firstLen) ^ seed)
	if keyLen > minMatch {
		tailSeed := bits.RotateLeft64(seed, 17)
		tail := mixLongBucketKey(bytesToU64LE(suffix[minMatch:], keyLen-minMatch) ^ tailSeed)
		key ^= bits.RotateLeft64(tail, 23)
	}
	return mixLongBucketKey(key ^ uint64(keyLen)*0x9e3779b97f4a7c15)
}

func longBucketGroupHash(key uint64, keyLen int) uint64 {
	return mixLongBucketKey(key ^ uint64(keyLen)*0x9e3779b97f4a7c15)
}

func (b *longBucket) appendCandidate(candidate longBucketCandidate) {
	if len(b.candidates) == cap(b.candidates) {
		nextCapacity := 0
		if len(b.candidates) == longBucketIndexThreshold {
			// The first group table adds two index-threshold windows. Reserve that
			// same construction window for candidates so it does not immediately
			// copy again after the index becomes active.
			nextCapacity = len(b.candidates) + initialLongBucketGroupSlotCount
		} else if len(b.candidates) >= longBucketIndexThreshold {
			// Indexed buckets append only; doubling avoids retaining the runtime's
			// intermediate backing arrays as a large bucket is trained.
			nextCapacity = cap(b.candidates) * 2
		}
		if nextCapacity != 0 {
			next := make([]longBucketCandidate, len(b.candidates), nextCapacity)
			copy(next, b.candidates)
			b.candidates = next
		}
	}
	b.candidates = append(b.candidates, candidate)
}

// longBucketGroupStartSlot maps a mixed hash into an arbitrary-sized table.
// This permits a larger resize step without retaining power-of-two tables.
func longBucketGroupStartSlot(key uint64, keyLen, slotCount int) int {
	high, _ := bits.Mul64(longBucketGroupHash(key, keyLen), uint64(slotCount))
	return int(high)
}

func (b *longBucket) groupSlot(key uint64, keyLen int) int {
	slot := longBucketGroupStartSlot(key, keyLen, len(b.groupSlots))
	for {
		entry := b.groupSlots[slot]
		if entry == 0 {
			return slot
		}
		candidate := &b.candidates[entry-1]
		if suffixKeyLen(candidate.suffixLen) == keyLen && candidate.key == key {
			return slot
		}
		slot++
		if slot == len(b.groupSlots) {
			slot = 0
		}
	}
}

func (b *longBucket) enablePrevious() {
	if b.previous != nil {
		return
	}
	b.previous = make([]uint16, len(b.candidates), cap(b.candidates))
	for i := range b.previous {
		b.previous[i] = noLongBucketCandidate
	}
}

func (b *longBucket) buildGroupSlots(dictionary []byte, seed uint64) {
	b.groupSlots = make([]uint16, initialLongBucketGroupSlotCount)
	b.groupCount = 0
	b.keyLenMask = 0
	for i := range b.candidates {
		candidate := &b.candidates[i]
		start := int(candidate.dictStart)
		end := start + int(candidate.suffixLen)
		candidate.key = longBucketGroupKey(dictionary[start:end], seed)
		keyLen := suffixKeyLen(candidate.suffixLen)
		slot := b.groupSlot(candidate.key, keyLen)
		previous := b.groupSlots[slot]
		if previous != 0 {
			b.enablePrevious()
			b.previous[i] = previous - 1
		} else {
			b.groupCount++
		}
		b.groupSlots[slot] = uint16(i + 1)
		b.keyLenMask |= 1 << uint(keyLen)
	}
}

func (b *longBucket) growGroupSlots() {
	oldSlots := b.groupSlots
	// Resizing at 7/8 load into 5/2 times as many slots starts the new table
	// below 35% load, avoiding the intermediate allocation ladder.
	newSlotCount := len(oldSlots) * longBucketGroupGrowthNumerator / longBucketGroupGrowthDenominator
	b.groupSlots = make([]uint16, newSlotCount)
	for i := range b.candidates {
		candidate := &b.candidates[i]
		keyLen := suffixKeyLen(candidate.suffixLen)
		slot := b.groupSlot(candidate.key, keyLen)
		b.groupSlots[slot] = uint16(i + 1)
	}
}

// appendUnindexed keeps the pre-image's descending-length insertion and
// struct-of-arrays layout until the default bucket reaches the fixed index
// threshold.
func (b *longBucket) appendUnindexed(head uint64, suffixLen uint16, id uint16, dictStart uint32) {
	b.heads = append(b.heads, head)
	b.suffixLens = append(b.suffixLens, suffixLen)
	b.ids = append(b.ids, id)
	b.dictStarts = append(b.dictStarts, dictStart)
	for i := len(b.heads) - 1; i > 0; i-- {
		if b.suffixLens[i] <= b.suffixLens[i-1] {
			break
		}
		b.heads[i], b.heads[i-1] = b.heads[i-1], b.heads[i]
		b.suffixLens[i], b.suffixLens[i-1] = b.suffixLens[i-1], b.suffixLens[i]
		b.ids[i], b.ids[i-1] = b.ids[i-1], b.ids[i]
		b.dictStarts[i], b.dictStarts[i-1] = b.dictStarts[i-1], b.dictStarts[i]
	}
}

func (b *longBucket) enableIndexed(dictionary []byte, seed uint64) {
	b.candidates = make([]longBucketCandidate, len(b.heads), len(b.heads)+initialLongBucketGroupSlotCount)
	for i := range b.candidates {
		b.candidates[i] = longBucketCandidate{
			dictStart: b.dictStarts[i],
			suffixLen: b.suffixLens[i],
			id:        b.ids[i],
		}
	}
	b.heads = nil
	b.suffixLens = nil
	b.ids = nil
	b.dictStarts = nil
	b.buildGroupSlots(dictionary, seed)
}

func (b *longBucket) appendEntry(suffix []byte, suffixLen uint16, id uint16, dictStart uint32, dictionary []byte, seed uint64, indexEnabled bool) {
	if b.candidates == nil {
		headLen := len(suffix)
		if headLen > minMatch {
			headLen = minMatch
		}
		b.appendUnindexed(bytesToU64LE(suffix, headLen), suffixLen, id, dictStart)
		if indexEnabled && len(b.heads) == longBucketIndexThreshold {
			b.enableIndexed(dictionary, seed)
		}
		return
	}

	keyLen := suffixKeyLen(suffixLen)
	candidate := longBucketCandidate{
		key:       longBucketGroupKey(suffix, seed),
		dictStart: dictStart,
		suffixLen: suffixLen,
		id:        id,
	}
	slot := b.groupSlot(candidate.key, keyLen)
	existingGroup := b.groupSlots[slot] != 0
	if !existingGroup && (b.groupCount+1)*longBucketGroupLoadDenominator > len(b.groupSlots)*longBucketGroupLoadNumerator {
		// Only a new group consumes a slot. Rebuild the accumulated entries
		// before appending it, so it cannot be linked to an old group.
		b.growGroupSlots()
		slot = b.groupSlot(candidate.key, keyLen)
	}

	candidateSlot := len(b.candidates)
	b.appendCandidate(candidate)
	if b.previous != nil {
		b.previous = append(b.previous, noLongBucketCandidate)
	}
	b.keyLenMask |= 1 << uint(keyLen)

	previous := b.groupSlots[slot]
	if previous != 0 {
		b.enablePrevious()
		b.previous[candidateSlot] = previous - 1
	} else if !existingGroup {
		b.groupCount++
	}
	b.groupSlots[slot] = uint16(candidateSlot + 1)
}

func (b *longBucket) lookupGroupEnd(key uint64, keyLen int) int {
	slot := b.groupSlot(key, keyLen)
	if entry := b.groupSlots[slot]; entry != 0 {
		return int(entry - 1)
	}
	return -1
}

// newMatcher creates a new empty longest prefix matcher.
func newMatcher(maxTokenLen int) *matcher {
	onPair16 := maxTokenLen == 16
	bucketSizeLimit := 0
	if onPair16 {
		bucketSizeLimit = maxOnPair16BucketSize
	}

	return &matcher{
		endPositions:    []uint32{0},
		onPair16:        onPair16,
		bucketSizeLimit: bucketSizeLimit,
		groupSeed:       newLongBucketGroupSeed(),
	}
}

// insert inserts a new pattern with associated token ID.
//
// Automatically chooses storage strategy based on pattern length:
// - Short patterns (≤8 bytes): Direct hash table insertion
// - Long patterns (>8 bytes): Bucketed by 8-byte prefix with suffix storage
//
// Long pattern buckets are kept sorted by pattern length (descending) for
// efficient longest-match-first lookup during matching.
//
// IMPORTANT: Token IDs must be inserted sequentially starting from 0!
func (m *matcher) insert(entry []byte, id uint16) bool {
	if len(entry) > minMatch {
		if len(entry)-minMatch > math.MaxUint16 {
			// suffixLens stores uint16; a longer pattern would silently
			// truncate and corrupt greedy matching, so reject it instead.
			return false
		}
		// Long pattern: store 8-byte prefix in bucket, suffix in dictionary
		prefix := bytesToU64LE(entry, minMatch)
		bucket := m.longMatchBuckets.get(prefix)
		if bucket != nil && m.bucketSizeLimit > 0 && bucket.len() >= m.bucketSizeLimit {
			return false
		}
		if bucket == nil {
			bucket = &longBucket{}
			m.longMatchBuckets.set(prefix, bucket)
		}

		suffix := entry[minMatch:]
		suffixLen := len(suffix)
		dictStart := uint32(len(m.dictionary))

		m.dictionary = append(m.dictionary, suffix...)
		m.endPositions = append(m.endPositions, uint32(len(m.dictionary)))
		bucket.appendEntry(suffix, uint16(suffixLen), id, dictStart, m.dictionary, m.groupSeed, m.bucketSizeLimit == 0)

		if m.longBits2 == nil {
			m.longBits2 = new([1024]uint64)
		}
		p2 := uint16(prefix)
		m.longBits2[p2>>6] |= 1 << (p2 & 63)

	} else {
		// Single-byte tokens are always byte-value identity tokens.
		if len(entry) == 1 {
			m.endPositions = append(m.endPositions, uint32(len(m.dictionary)))
			return true
		}

		// Short pattern: direct hash table lookup
		prefix := bytesToU64LE(entry, len(entry))
		m.shortMatchLookup[len(entry)].set(prefix, id)
		if m.lengthByPrefix2 == nil {
			m.lengthByPrefix2 = new([65536]uint8)
		}
		m.lengthByPrefix2[uint16(prefix)] |= 1 << uint(len(entry))
		m.endPositions = append(m.endPositions, uint32(len(m.dictionary)))
	}
	return true
}

// find finds the longest matching pattern for the given input data.
//
// Returns the token ID and match length for the longest pattern that matches
// the beginning of the input data. Uses two-phase search:
//
// 1. Long pattern search: Check bucketed patterns (>8 bytes) first for longest matches
// 2. Short pattern search: Check direct lookup patterns (≤8 bytes) in decreasing length order
func (m *matcher) find(data []byte) (uint16, int, bool) {
	// The first up-to-8 bytes serve as both the long-bucket prefix key and the
	// short-lookup probe window, so load them once.
	maxLen := minMatch
	if len(data) < maxLen {
		maxLen = len(data)
	}
	low8 := bytesToU64LE(data, maxLen)

	// Phase 1: Long pattern search (>8 bytes) - check longest matches first.
	// Gate table access behind a 2-byte-prefix bitset so non-matching inputs
	// skip the table probe entirely.
	if len(data) > minMatch && m.longBits2 != nil {
		p2 := uint16(low8)
		if m.longBits2[p2>>6]&(1<<(p2&63)) != 0 {
			inputSuffix := data[minMatch:]
			if bucket := m.longMatchBuckets.get(low8); bucket != nil {
				if bucket.candidates == nil {
					// Below the fixed index threshold retain the pre-image's
					// descending-length struct-of-arrays scan.
					inputHeadLen := len(inputSuffix)
					if inputHeadLen > minMatch {
						inputHeadLen = minMatch
					}
					inputHead := bytesToU64LE(inputSuffix, inputHeadLen)
					heads := bucket.heads
					lens := bucket.suffixLens
					for i := 0; i < len(heads); i++ {
						sLen := int(lens[i])
						if sLen > len(inputSuffix) {
							continue
						}
						matchLen := sLen
						if matchLen > minMatch {
							matchLen = minMatch
						}
						if (heads[i]^inputHead)&masks[matchLen] != 0 {
							continue
						}
						if sLen <= minMatch {
							return bucket.ids[i], minMatch + sLen, true
						}
						start := int(bucket.dictStarts[i])
						if bytes.Equal(m.dictionary[start+minMatch:start+sLen], inputSuffix[minMatch:sLen]) {
							return bucket.ids[i], minMatch + sLen, true
						}
					}
				} else {
					inputKeyLen := len(inputSuffix)
					if inputKeyLen > longBucketGroupKeyBytes {
						inputKeyLen = longBucketGroupKeyBytes
					}
					matchingKeyLens := bucket.keyLenMask
					matchingKeyLens &= (uint32(1) << uint(inputKeyLen+1)) - 1
					for matchingKeyLens != 0 {
						keyLen := bits.Len32(matchingKeyLens) - 1
						key := longBucketGroupKey(inputSuffix[:keyLen], m.groupSeed)
						candidateEnd := bucket.lookupGroupEnd(key, keyLen)
						if candidateEnd >= 0 {
							bestLen := -1
							var bestID uint16
							for i := candidateEnd; i != int(noLongBucketCandidate); {
								candidate := &bucket.candidates[i]
								sLen := int(candidate.suffixLen)
								if sLen <= len(inputSuffix) {
									start := int(candidate.dictStart)
									if bytes.Equal(m.dictionary[start:start+sLen], inputSuffix[:sLen]) &&
										(sLen > bestLen || (sLen == bestLen && candidate.id < bestID)) {
										bestLen = sLen
										bestID = candidate.id
									}
								}
								if bucket.previous == nil {
									break
								}
								i = int(bucket.previous[i])
							}
							if bestLen >= 0 {
								return bestID, minMatch + bestLen, true
							}
						}
						matchingKeyLens &^= 1 << uint(keyLen)
					}
				}
			}
		}
	}

	// Phase 2: Short pattern search (≤8 bytes) - longest to shortest.
	// Use 2-byte-prefix bitmask to skip lengths with no candidates.
	if maxLen >= 2 && m.lengthByPrefix2 != nil {
		lenMask := m.lengthByPrefix2[uint16(low8)]
		// Drop bits for lengths > maxLen. maxLen ≤ 8 so mask fits in uint8.
		lenMask &= (1 << (uint(maxLen) + 1)) - 1
		for lenMask != 0 {
			length := bits.Len8(lenMask) - 1
			if id, ok := m.shortMatchLookup[length].get(low8 & masks[length]); ok {
				return id, length, true
			}
			lenMask &^= 1 << length
		}
	}
	if len(data) > 0 {
		return uint16(data[0]), 1, true
	}

	return 0, 0, false
}

// bytesToU64LE converts byte sequence to little-endian u64 with length masking.
func bytesToU64LE(bytes []byte, length int) uint64 {
	// Clamp length to valid range
	if length > 8 {
		length = 8
	}
	if length < 0 {
		length = 0
	}

	if len(bytes) < 8 {
		// Safe path for short slices
		var buf [8]byte
		copy(buf[:], bytes)
		value := binary.LittleEndian.Uint64(buf[:])
		return value & masks[length]
	}

	// Fast path using unsafe pointer
	// Safe because we verified len(bytes) >= 8 above
	ptr := unsafe.Pointer(&bytes[0])
	value := *(*uint64)(ptr)
	return value & masks[length]
}
