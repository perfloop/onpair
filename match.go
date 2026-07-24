package onpair

import (
	"bytes"
	"encoding/binary"
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
}

// longBucket keeps small buckets ordered in place. Larger default buckets use
// an open-addressed suffix-head index whose linked groups are checked without
// scanning unrelated heads.
type longBucket struct {
	candidates  []longBucketCandidate
	previous    []uint16 // prior candidate in the same suffix-head group, or noLongBucketCandidate
	groupSlots  []uint16 // open-addressed group key → latest candidate slot; zero means empty
	groupCount  int
	headLenMask uint16 // bit L set => a group with min(suffixLen, 8) == L exists
}

const (
	noLongBucketCandidate            = ^uint16(0)
	longBucketIndexThreshold         = maxOnPair16BucketSize
	initialLongBucketGroupSlotCount  = 2 * maxOnPair16BucketSize
	longBucketGroupLoadNumerator     = 7
	longBucketGroupLoadDenominator   = 8
	longBucketGroupGrowthNumerator   = 5
	longBucketGroupGrowthDenominator = 2
)

type longBucketCandidate struct {
	head      uint64 // first min(suffixLen, 8) bytes of suffix, LE, zero-extended
	dictStart uint32 // offset in m.dictionary where this suffix starts
	suffixLen uint16 // bytes past the 8-byte prefix
	id        uint16
}

func (b *longBucket) len() int { return len(b.candidates) }

func suffixHeadLen(suffixLen uint16) int {
	headLen := int(suffixLen)
	if headLen > minMatch {
		return minMatch
	}
	return headLen
}

func longBucketGroupHash(head uint64, headLen int) uint64 {
	// The suffix head is attacker supplied, so mix both the packed bytes and
	// length class before probing the open-addressed group table.
	head ^= uint64(headLen) * 0x9e3779b97f4a7c15
	head ^= head >> 30
	head *= 0xbf58476d1ce4e5b9
	head ^= head >> 27
	head *= 0x94d049bb133111eb
	return head ^ (head >> 31)
}

func (b *longBucket) appendCandidate(candidate longBucketCandidate) {
	if len(b.candidates) == cap(b.candidates) && len(b.candidates) == longBucketIndexThreshold {
		// The first group table adds two index-threshold windows. Reserve that
		// same construction window for candidates so it does not immediately
		// copy again after the index becomes active.
		next := make([]longBucketCandidate, len(b.candidates), len(b.candidates)+initialLongBucketGroupSlotCount)
		copy(next, b.candidates)
		b.candidates = next
	}
	b.candidates = append(b.candidates, candidate)
}

// longBucketGroupStartSlot maps a mixed hash into an arbitrary-sized table.
// This permits a larger resize step without retaining power-of-two tables.
func longBucketGroupStartSlot(head uint64, headLen, slotCount int) int {
	high, _ := bits.Mul64(longBucketGroupHash(head, headLen), uint64(slotCount))
	return int(high)
}

func (b *longBucket) groupSlot(head uint64, headLen int) int {
	slot := longBucketGroupStartSlot(head, headLen, len(b.groupSlots))
	for {
		entry := b.groupSlots[slot]
		if entry == 0 {
			return slot
		}
		candidate := &b.candidates[entry-1]
		if suffixHeadLen(candidate.suffixLen) == headLen && candidate.head == head {
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

func (b *longBucket) buildGroupSlots() {
	b.groupSlots = make([]uint16, initialLongBucketGroupSlotCount)
	b.groupCount = 0
	for i := range b.candidates {
		candidate := &b.candidates[i]
		headLen := suffixHeadLen(candidate.suffixLen)
		slot := b.groupSlot(candidate.head, headLen)
		previous := b.groupSlots[slot]
		if previous != 0 {
			b.enablePrevious()
			b.previous[i] = previous - 1
		} else {
			b.groupCount++
		}
		b.groupSlots[slot] = uint16(i + 1)
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
		headLen := suffixHeadLen(candidate.suffixLen)
		slot := b.groupSlot(candidate.head, headLen)
		b.groupSlots[slot] = uint16(i + 1)
	}
}

func (b *longBucket) appendOrdered(candidate longBucketCandidate, headLen int) {
	// The public 16-byte mode cannot exceed this bounded window. Keeping it
	// ordered retains its existing greedy lookup without allocating an index.
	lo, hi := 0, len(b.candidates)
	for lo < hi {
		mid := lo + (hi-lo)/2
		current := &b.candidates[mid]
		currentHeadLen := suffixHeadLen(current.suffixLen)
		if currentHeadLen < headLen ||
			(currentHeadLen == headLen && (current.head < candidate.head ||
				(current.head == candidate.head && current.suffixLen >= candidate.suffixLen))) {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	b.candidates = append(b.candidates, longBucketCandidate{})
	copy(b.candidates[lo+1:], b.candidates[lo:])
	b.candidates[lo] = candidate
	b.headLenMask |= 1 << uint(headLen)
}

// hasOnlyGroup is valid while candidates retain their pre-index sort order.
func (b *longBucket) hasOnlyGroup(head uint64, headLen int) bool {
	if len(b.candidates) == 0 {
		return false
	}
	first := &b.candidates[0]
	last := &b.candidates[len(b.candidates)-1]
	return suffixHeadLen(first.suffixLen) == headLen && first.head == head &&
		suffixHeadLen(last.suffixLen) == headLen && last.head == head
}

func (b *longBucket) appendEntry(head uint64, suffixLen uint16, id uint16, dictStart uint32) {
	headLen := suffixHeadLen(suffixLen)
	head &= masks[headLen]
	candidate := longBucketCandidate{
		head:      head,
		dictStart: dictStart,
		suffixLen: suffixLen,
		id:        id,
	}
	if len(b.candidates) < longBucketIndexThreshold {
		b.appendOrdered(candidate, headLen)
		return
	}
	if b.groupSlots == nil && b.hasOnlyGroup(head, headLen) {
		// A single suffix-head group has no unrelated heads to skip. Retain its
		// existing length order without allocating index or linkage storage.
		if suffixLen <= b.candidates[len(b.candidates)-1].suffixLen {
			b.appendCandidate(candidate)
			b.headLenMask |= 1 << uint(headLen)
			return
		}
		b.appendOrdered(candidate, headLen)
		return
	}
	slot, existingGroup := 0, false
	if b.groupSlots != nil {
		slot = b.groupSlot(head, headLen)
		existingGroup = b.groupSlots[slot] != 0
		if !existingGroup && (b.groupCount+1)*longBucketGroupLoadDenominator > len(b.groupSlots)*longBucketGroupLoadNumerator {
			// Only a new group consumes a slot. Rebuild the accumulated entries
			// before appending it, so it cannot be linked to an old group.
			b.growGroupSlots()
			slot = b.groupSlot(head, headLen)
		}
	}
	candidateSlot := len(b.candidates)
	b.appendCandidate(candidate)
	if b.previous != nil {
		b.previous = append(b.previous, noLongBucketCandidate)
	}
	b.headLenMask |= 1 << uint(headLen)

	if b.groupSlots == nil && len(b.candidates) >= longBucketIndexThreshold {
		b.buildGroupSlots()
		return
	}
	if b.groupSlots == nil {
		return
	}
	previous := b.groupSlots[slot]
	if previous != 0 {
		b.enablePrevious()
		b.previous[candidateSlot] = previous - 1
	} else if !existingGroup {
		b.groupCount++
	}
	b.groupSlots[slot] = uint16(candidateSlot + 1)
}

func (b *longBucket) lookupGroupStart(head uint64, headLen int) int {
	head &= masks[headLen]
	lo, hi := 0, len(b.candidates)
	for lo < hi {
		mid := lo + (hi-lo)/2
		candidate := &b.candidates[mid]
		candidateHeadLen := suffixHeadLen(candidate.suffixLen)
		if candidateHeadLen < headLen || (candidateHeadLen == headLen && candidate.head < head) {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo == len(b.candidates) {
		return -1
	}
	candidate := &b.candidates[lo]
	if suffixHeadLen(candidate.suffixLen) != headLen || candidate.head != head {
		return -1
	}
	return lo
}

func (b *longBucket) lookupGroupEnd(head uint64, headLen int) int {
	if b.groupSlots == nil {
		return -1
	}
	head &= masks[headLen]
	slot := b.groupSlot(head, headLen)
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
		headLen := suffixLen
		if headLen > minMatch {
			headLen = minMatch
		}
		head := bytesToU64LE(suffix, headLen)
		dictStart := uint32(len(m.dictionary))

		m.dictionary = append(m.dictionary, suffix...)
		m.endPositions = append(m.endPositions, uint32(len(m.dictionary)))
		bucket.appendEntry(head, uint16(suffixLen), id, dictStart)

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
			inputHeadLen := len(inputSuffix)
			if inputHeadLen > minMatch {
				inputHeadLen = minMatch
			}
			inputHead := bytesToU64LE(inputSuffix, inputHeadLen)

			if bucket := m.longMatchBuckets.get(low8); bucket != nil {
				matchingHeadLens := bucket.headLenMask
				matchingHeadLens &= (1 << (uint(inputHeadLen) + 1)) - 1
				for matchingHeadLens != 0 {
					headLen := bits.Len16(matchingHeadLens) - 1
					if bucket.groupSlots != nil {
						candidateEnd := bucket.lookupGroupEnd(inputHead, headLen)
						if candidateEnd >= 0 {
							bestLen := -1
							var bestID uint16
							for i := candidateEnd; i != int(noLongBucketCandidate); {
								candidate := &bucket.candidates[i]
								sLen := int(candidate.suffixLen)
								if sLen <= len(inputSuffix) {
									matched := sLen <= minMatch
									if !matched {
										// Suffix longer than 8 bytes: verify its tail.
										start := int(candidate.dictStart)
										matched = bytes.Equal(m.dictionary[start+minMatch:start+sLen], inputSuffix[minMatch:sLen])
									}
									if matched && (sLen > bestLen || (sLen == bestLen && candidate.id < bestID)) {
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
					} else if bucket.hasOnlyGroup(inputHead&masks[headLen], headLen) {
						for i := range bucket.candidates {
							candidate := &bucket.candidates[i]
							sLen := int(candidate.suffixLen)
							if sLen > len(inputSuffix) {
								continue
							}
							if sLen <= minMatch {
								return candidate.id, minMatch + sLen, true
							}
							start := int(candidate.dictStart)
							if bytes.Equal(m.dictionary[start+minMatch:start+sLen], inputSuffix[minMatch:sLen]) {
								return candidate.id, minMatch + sLen, true
							}
						}
					} else if candidateStart := bucket.lookupGroupStart(inputHead, headLen); candidateStart >= 0 {
						groupHead := inputHead & masks[headLen]
						for i := candidateStart; i < len(bucket.candidates); i++ {
							candidate := &bucket.candidates[i]
							if suffixHeadLen(candidate.suffixLen) != headLen || candidate.head != groupHead {
								break
							}
							sLen := int(candidate.suffixLen)
							if sLen > len(inputSuffix) {
								continue
							}
							if sLen <= minMatch {
								return candidate.id, minMatch + sLen, true
							}
							start := int(candidate.dictStart)
							if bytes.Equal(m.dictionary[start+minMatch:start+sLen], inputSuffix[minMatch:sLen]) {
								return candidate.id, minMatch + sLen, true
							}
						}
					}
					matchingHeadLens &^= 1 << uint(headLen)
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
