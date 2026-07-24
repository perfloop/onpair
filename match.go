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

// longBucket keeps a secondary index ordered by the packed suffix head and
// its length class. The matching group is contiguous, with longer suffixes
// first, so lookup remains greedy without scanning unrelated heads.
type longBucket struct {
	candidates  []longBucketCandidate
	headLenMask uint16 // bit L set => a group with min(suffixLen, 8) == L exists
}

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

func (b *longBucket) appendEntry(head uint64, suffixLen uint16, id uint16, dictStart uint32) {
	headLen := suffixHeadLen(suffixLen)
	head &= masks[headLen]

	// Keep candidates ordered by (head length, head, descending suffix length).
	// Equal-length entries remain stable, matching the old insertion sort.
	lo, hi := 0, len(b.candidates)
	for lo < hi {
		mid := lo + (hi-lo)/2
		candidate := &b.candidates[mid]
		candidateHeadLen := suffixHeadLen(candidate.suffixLen)
		if candidateHeadLen < headLen ||
			(candidateHeadLen == headLen && (candidate.head < head ||
				(candidate.head == head && candidate.suffixLen >= suffixLen))) {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	b.candidates = append(b.candidates, longBucketCandidate{})
	copy(b.candidates[lo+1:], b.candidates[lo:])
	b.candidates[lo] = longBucketCandidate{
		head:      head,
		dictStart: dictStart,
		suffixLen: suffixLen,
		id:        id,
	}
	b.headLenMask |= 1 << uint(headLen)
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
					candidateStart := bucket.lookupGroupStart(inputHead, headLen)
					if candidateStart >= 0 {
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
							// Suffix longer than 8 bytes: verify the tail past the head.
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
