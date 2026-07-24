package onpair

import (
	"bytes"
	"math"
	"testing"
)

func TestMatcherInsertRejectsOversizedPattern(t *testing.T) {
	// suffixLens stores uint16: a pattern whose suffix exceeds 65535 bytes
	// must be rejected, not silently truncated (regression: truncation made
	// greedy matching advance by the wrapped length, corrupting round-trips).
	m := newMatcher(0)
	limit := make([]byte, minMatch+math.MaxUint16)
	if !m.insert(limit, 256) {
		t.Fatal("pattern at the representable limit should insert")
	}
	over := make([]byte, minMatch+math.MaxUint16+1)
	if m.insert(over, 257) {
		t.Fatal("pattern past the representable limit must be rejected")
	}
}

func TestRoundTripHighlyRepetitiveDefaultConfig(t *testing.T) {
	// Threshold 1 doubles token lengths aggressively; before the insert guard
	// this crossed the 65,543-byte representable limit and round-trips came
	// back with the wrong length.
	input := string(bytes.Repeat([]byte{'a'}, 512*1024))
	archive := mustEncode(NewEncoder(WithThreshold(1)), []string{input})
	out, err := archive.AppendRow(nil, 0)
	if err != nil {
		t.Fatalf("AppendRow: %v", err)
	}
	if string(out) != input {
		t.Fatalf("round-trip mismatch: got %d bytes, want %d", len(out), len(input))
	}
}

func TestOnPair16MatcherBucketBound(t *testing.T) {
	m := newMatcher(16)
	prefix := []byte("abcdefgh")
	prefixKey := bytesToU64LE(prefix, minMatch)

	inserted := 0
	for i := 0; i < maxOnPair16BucketSize+32; i++ {
		entry := append([]byte(nil), prefix...)
		entry = append(entry, byte(i), byte(i>>8))
		if m.insert(entry, uint16(inserted)) {
			inserted++
		}
	}

	if inserted != maxOnPair16BucketSize {
		t.Fatalf("inserted long-token count mismatch: got %d want %d", inserted, maxOnPair16BucketSize)
	}
	if got := m.longMatchBuckets.get(prefixKey).len(); got != maxOnPair16BucketSize {
		t.Fatalf("bucket size mismatch: got %d want %d", got, maxOnPair16BucketSize)
	}
}

func TestOnPair16MatcherFindLongToken(t *testing.T) {
	m := newMatcher(16)
	token := []byte("abcdefghXYZ")

	if !m.insert(token, 0) {
		t.Fatalf("insert failed")
	}
	id, n, ok := m.find([]byte("abcdefghXYZ_tail"))
	if !ok {
		t.Fatalf("expected long match")
	}
	if id != 0 {
		t.Fatalf("token id mismatch: got %d want 0", id)
	}
	if n != len(token) {
		t.Fatalf("token length mismatch: got %d want %d", n, len(token))
	}
}

func TestLongBucketRepeatedGroupMatchesReference(t *testing.T) {
	const prefix = "abcdefgh"
	const sharedHead = "HEADHEAD"

	matcher := newMatcher(0)
	tokens := make([][]byte, 0, 551)
	insert := func(suffix []byte) {
		token := append([]byte(prefix), suffix...)
		if !matcher.insert(token, uint16(len(tokens))) {
			t.Fatalf("insert token %d", len(tokens))
		}
		tokens = append(tokens, token)
	}

	// The 8-byte suffix is a valid fallback. Every later token shares its
	// suffix-head group but has a distinct tail and a range of total lengths.
	insert([]byte(sharedHead))
	for i := 0; i < 550; i++ {
		suffix := make([]byte, len(sharedHead)+3+i%19)
		copy(suffix, sharedHead)
		for j := len(sharedHead); j < len(suffix)-2; j++ {
			suffix[j] = byte('a' + (i+j*17)%26)
		}
		suffix[len(suffix)-2] = byte(i)
		suffix[len(suffix)-1] = byte(i >> 8)
		insert(suffix)
	}

	wantMatch := func(input []byte) (uint16, int) {
		bestID, bestLen := uint16(0), -1
		for id, token := range tokens {
			if len(token) > bestLen && bytes.HasPrefix(input, token) {
				bestID, bestLen = uint16(id), len(token)
			}
		}
		return bestID, bestLen
	}
	probes := make([][]byte, 0, 80)
	for i := 0; i < len(tokens); i += 7 {
		probes = append(probes, append(append([]byte(nil), tokens[i]...), 0xff))
	}
	rejectProbe := append([]byte(prefix+sharedHead), 0x00, 0xff, 0x80)
	wantID, wantLen := wantMatch(rejectProbe)
	if wantID != 0 || wantLen != len(prefix)+len(sharedHead) {
		t.Fatalf("reference reject probe: got id=%d len=%d", wantID, wantLen)
	}
	probes = append(probes, rejectProbe)
	for _, probe := range probes {
		wantID, wantLen := wantMatch(probe)
		gotID, gotLen, ok := matcher.find(probe)
		if !ok || gotID != wantID || gotLen != wantLen {
			t.Fatalf("find(%x): got id=%d len=%d ok=%v; want id=%d len=%d", probe, gotID, gotLen, ok, wantID, wantLen)
		}
	}
}

func TestLongBucketIndexedGroupsMatchReference(t *testing.T) {
	const prefix = "abcdefgh"
	const groups = 700

	matcher := newMatcher(0)
	tokens := make([][]byte, 0, groups)
	for i := 0; i < groups; i++ {
		suffix := []byte{
			byte(i), byte(i >> 8), byte(i >> 16), byte(i >> 24),
			0x91, 0x92, 0x93, 0x94, byte(i ^ 0x55), byte(i >> 3),
		}
		token := append([]byte(prefix), suffix...)
		if !matcher.insert(token, uint16(i)) {
			t.Fatalf("insert token %d", i)
		}
		tokens = append(tokens, token)
	}

	bucket := matcher.longMatchBuckets.get(bytesToU64LE([]byte(prefix), minMatch))
	if bucket == nil || bucket.groupSlots == nil {
		t.Fatal("distinct suffix heads did not build an index")
	}
	for i := 0; i < len(tokens); i += 17 {
		probe := append(append([]byte(nil), tokens[i]...), 0xff)
		gotID, gotLen, ok := matcher.find(probe)
		if !ok || gotID != uint16(i) || gotLen != len(tokens[i]) {
			t.Fatalf("find group %d: got id=%d len=%d ok=%v; want id=%d len=%d", i, gotID, gotLen, ok, i, len(tokens[i]))
		}
	}
}
