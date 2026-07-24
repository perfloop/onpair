package onpair

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/bits"
	"sort"
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
	longMatchBuckets longBucketTable // 8-byte prefix → candidate bucket
	shortMatchLookup [9]u64U16Table  // length → (prefix, token ID)
	lengthByPrefix2  *[65536]uint8   // bit L set ⇒ some short token of length L starts with these 2 LE bytes
	longBits2        *[1024]uint64   // bit set ⇒ some long token starts with these 2 LE bytes (65536-bit set)
	dictionary       []byte          // Suffix storage for long patterns
	onPair16         bool
	bucketSizeLimit  int
}

// longBucket is a struct-of-arrays layout with append-only entry storage.
// heads + suffixLens are the hot arrays touched on every probe (reject path);
// ids + dictStarts are cold, read only when the packed head matches and the
// suffix exceeds 8 bytes. Finalization sorts the payload arrays once after
// training, so parsing retains direct longest-first traversal.
type longBucket struct {
	heads      []uint64 // first min(suffixLen, 8) bytes of suffix, LE, zero-extended
	suffixLens []uint16 // bytes past the 8-byte prefix; token length = suffixLen + 8
	ids        []uint16
	dictStarts []uint32 // offset in m.dictionary where this suffix starts
}

func (b *longBucket) len() int { return len(b.heads) }
func (b *longBucket) Len() int { return len(b.heads) }

func (b *longBucket) appendEntry(head uint64, suffixLen uint16, id uint16, dictStart uint32) {
	b.heads = append(b.heads, head)
	b.suffixLens = append(b.suffixLens, suffixLen)
	b.ids = append(b.ids, id)
	b.dictStarts = append(b.dictStarts, dictStart)
}

func (b *longBucket) Less(i, j int) bool { return b.suffixLens[i] > b.suffixLens[j] }

func (b *longBucket) Swap(i, j int) {
	b.heads[i], b.heads[j] = b.heads[j], b.heads[i]
	b.suffixLens[i], b.suffixLens[j] = b.suffixLens[j], b.suffixLens[i]
	b.ids[i], b.ids[j] = b.ids[j], b.ids[i]
	b.dictStarts[i], b.dictStarts[j] = b.dictStarts[j], b.dictStarts[i]
}

func (b *longBucket) sortBySuffixLen() {
	switch len(b.heads) {
	case 0, 1:
		return
	case 2:
		if b.Less(1, 0) {
			b.Swap(0, 1)
		}
		return
	}

	needsReorder := false
	for i := 1; i < len(b.suffixLens); i++ {
		if b.suffixLens[i] > b.suffixLens[i-1] {
			needsReorder = true
			break
		}
	}
	if !needsReorder {
		return
	}

	// sort.Sort uses longBucket.Swap to move the aligned payload arrays in
	// place. Unlike the temporary permutation, it adds no per-bucket storage.
	sort.Sort(b)
}

func (m *matcher) finalizeLongBuckets() {
	for i := range m.longMatchBuckets.entries {
		bucket := m.longMatchBuckets.entries[i].bucket
		if bucket != nil {
			bucket.sortBySuffixLen()
		}
	}
}

// newMatcher creates a new empty longest prefix matcher.
func newMatcher(maxTokenLen int) *matcher {
	onPair16 := maxTokenLen == 16
	bucketSizeLimit := 0
	if onPair16 {
		bucketSizeLimit = maxOnPair16BucketSize
	}

	return &matcher{
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
		bucket.appendEntry(head, uint16(suffixLen), id, dictStart)

		if m.longBits2 == nil {
			m.longBits2 = new([1024]uint64)
		}
		p2 := uint16(prefix)
		m.longBits2[p2>>6] |= 1 << (p2 & 63)

	} else {
		// Single-byte tokens are always byte-value identity tokens.
		if len(entry) == 1 {
			return true
		}

		// Short pattern: direct hash table lookup
		prefix := bytesToU64LE(entry, len(entry))
		m.shortMatchLookup[len(entry)].set(prefix, id)
		if m.lengthByPrefix2 == nil {
			m.lengthByPrefix2 = new([65536]uint8)
		}
		m.lengthByPrefix2[uint16(prefix)] |= 1 << uint(len(entry))
	}
	return true
}

// findUnorderedLong preserves greedy matching while buildTokens has appended
// entries that finalizeLongBuckets has not yet reordered.
func (m *matcher) findUnorderedLong(bucket *longBucket, inputSuffix []byte, inputHead uint64) (uint16, int, bool) {
	heads := bucket.heads
	lens := bucket.suffixLens
	bestIndex := -1
	bestSuffixLen := 0
	for i := range heads {
		sLen := int(lens[i])
		if sLen > len(inputSuffix) {
			continue
		}
		mLen := sLen
		if mLen > minMatch {
			mLen = minMatch
		}
		if (heads[i]^inputHead)&masks[mLen] != 0 {
			continue
		}
		if sLen > minMatch {
			start := int(bucket.dictStarts[i])
			if !bytes.Equal(m.dictionary[start+minMatch:start+sLen], inputSuffix[minMatch:sLen]) {
				continue
			}
		}
		if sLen > bestSuffixLen {
			bestIndex = i
			bestSuffixLen = sLen
		}
	}
	if bestIndex < 0 {
		return 0, 0, false
	}
	return bucket.ids[bestIndex], minMatch + bestSuffixLen, true
}

// findShort finds the longest direct-table pattern or falls back to a byte.
func (m *matcher) findShort(data []byte, maxLen int, low8 uint64) (uint16, int, bool) {
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

// find finds the longest matching pattern for a finalized matcher. Its long
// buckets are contiguous longest-first payload arrays, so the Encode path can
// traverse them directly without a training-state branch.
func (m *matcher) find(data []byte) (uint16, int, bool) {
	maxLen := minMatch
	if len(data) < maxLen {
		maxLen = len(data)
	}
	low8 := bytesToU64LE(data, maxLen)

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
				heads := bucket.heads
				lens := bucket.suffixLens
				for i := range heads {
					sLen := int(lens[i])
					if sLen > len(inputSuffix) {
						continue
					}
					// Packed head prefilter: XOR the stored head with the input's
					// head masked to the relevant length. For suffixes ≤ 8 bytes
					// this is authoritative; otherwise it's a cheap reject.
					mLen := sLen
					if mLen > minMatch {
						mLen = minMatch
					}
					if (heads[i]^inputHead)&masks[mLen] != 0 {
						continue
					}
					if sLen <= minMatch {
						return bucket.ids[i], minMatch + sLen, true
					}
					// Suffix longer than 8 bytes: verify the tail past the head.
					start := int(bucket.dictStarts[i])
					if bytes.Equal(m.dictionary[start+minMatch:start+sLen], inputSuffix[minMatch:sLen]) {
						return bucket.ids[i], minMatch + sLen, true
					}
				}
			}
		}
	}

	// Phase 2: short patterns (≤8 bytes), longest first. Keep this direct in
	// the finalized parser path; buildTokens alone uses findShort below.
	if maxLen >= 2 && m.lengthByPrefix2 != nil {
		lenMask := m.lengthByPrefix2[uint16(low8)]
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

// findUnordered finds the longest matching pattern while buildTokens is still
// appending long-bucket entries. Encoder.train finalizes those arrays before
// the matcher reaches the Encode path.
func (m *matcher) findUnordered(data []byte) (uint16, int, bool) {
	maxLen := minMatch
	if len(data) < maxLen {
		maxLen = len(data)
	}
	low8 := bytesToU64LE(data, maxLen)

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
				if id, length, ok := m.findUnorderedLong(bucket, inputSuffix, inputHead); ok {
					return id, length, true
				}
			}
		}
	}
	return m.findShort(data, maxLen, low8)
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
