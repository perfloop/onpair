package onpair

import (
	"bytes"
	"testing"
)

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

	bucket := matcher.longMatchBuckets.get(bytesToU64LE([]byte(prefix), minMatch))
	if bucket == nil || bucket.groupSlots == nil || bucket.previous == nil {
		t.Fatal("repeated group did not build its secondary index")
	}
	if len(bucket.groupSlots) <= initialLongBucketGroupSlotCount {
		t.Fatalf("group table did not rehash: got %d slots", len(bucket.groupSlots))
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
