package onpair

import (
	"slices"
	"testing"
)

const (
	lowInversionLongBucketEntries = 2040
	lowInversionLeadingLongSuffix = 1019
)

var lowInversionLongBucketPrefix [minMatch]byte

func lowInversionSuffixes() []string {
	suffixes := make([]string, 0, lowInversionLongBucketEntries)
	threeByteSuffix := func(i int) string {
		return string([]byte{0xc1, byte(i), byte(i >> 8)})
	}
	twoByteSuffix := func(i int) string {
		return string([]byte{byte(i), byte(i >> 8)})
	}

	for i := 0; i < lowInversionLeadingLongSuffix; i++ {
		suffixes = append(suffixes, threeByteSuffix(i))
	}
	suffixes = append(suffixes, twoByteSuffix(0))
	suffixes = append(suffixes, threeByteSuffix(lowInversionLeadingLongSuffix))
	for i := 1; len(suffixes) < lowInversionLongBucketEntries; i++ {
		suffixes = append(suffixes, twoByteSuffix(i))
	}
	return suffixes
}

// lowInversionLongBucketRows makes the training shuffle consume three phases:
// the shared prefix, all suffix tokens, and then prefix+suffix rows. The long
// entries arrive as 1,019 length-3 suffixes, one length-2 suffix, one
// length-3 suffix, and the remaining length-2 suffixes. That is exactly one
// inversion of an otherwise non-increasing suffix-length order.
func lowInversionLongBucketRows() []string {
	suffixes := lowInversionSuffixes()
	phases := make([]string, 0, 1+2*len(suffixes))
	phases = append(phases, string(lowInversionLongBucketPrefix[:]))
	phases = append(phases, suffixes...)
	for _, suffix := range suffixes {
		row := make([]byte, minMatch+len(suffix))
		copy(row, lowInversionLongBucketPrefix[:])
		copy(row[minMatch:], suffix)
		phases = append(phases, string(row))
	}

	rows := make([]string, len(phases))
	for phase, index := range deterministicTrainingOrder(len(rows)) {
		rows[index] = phases[phase]
	}
	return rows
}

func unfinalizedLowInversionMatcher(rows []string) *matcher {
	data, endPositions := flattenStrings(rows)
	encoder := NewEncoder(WithThreshold(1))
	matcher := newMatcher(encoder.config.MaxTokenLen)
	dictionary := make([]byte, 0, 1024*1024)
	tokenBoundaries := make([]uint32, 0, singleByteTokens+4096)
	tokenBoundaries = append(tokenBoundaries, 0)
	for i := 0; i < singleByteTokens; i++ {
		_ = matcher.insert([]byte{byte(i)}, uint16(i))
		dictionary = append(dictionary, byte(i))
		tokenBoundaries = append(tokenBoundaries, uint32(len(dictionary)))
	}

	_, _ = encoder.buildTokens(
		data, endPositions, deterministicTrainingOrder(len(rows)), len(data),
		matcher, dictionary, tokenBoundaries,
		encoder.config.Threshold, resolveTokenLimit(encoder.config),
	)
	return matcher
}

func verifyLowInversionLongBucket(tb testing.TB, bucket *longBucket, finalized bool) {
	tb.Helper()
	if bucket == nil {
		tb.Fatal("missing low-inversion long bucket")
	}
	if got := bucket.len(); got != lowInversionLongBucketEntries {
		tb.Fatalf("low-inversion entries: got %d, want %d", got, lowInversionLongBucketEntries)
	}

	if finalized {
		for i := 1; i < len(bucket.suffixLens); i++ {
			if bucket.suffixLens[i] > bucket.suffixLens[i-1] {
				tb.Fatalf("finalized suffix lengths increase at %d: %d then %d", i, bucket.suffixLens[i-1], bucket.suffixLens[i])
			}
		}
		return
	}

	want := make([]uint16, 0, lowInversionLongBucketEntries)
	for i := 0; i < lowInversionLeadingLongSuffix; i++ {
		want = append(want, 3)
	}
	want = append(want, 2, 3)
	for len(want) < lowInversionLongBucketEntries {
		want = append(want, 2)
	}
	if !slices.Equal(bucket.suffixLens, want) {
		tb.Fatalf("unfinalized suffix lengths differ from the one-inversion fixture")
	}
}

func TestLongBucketTrainingPreservesLowInversionFixture(t *testing.T) {
	rows := lowInversionLongBucketRows()
	prefix := bytesToU64LE(lowInversionLongBucketPrefix[:], minMatch)

	unfinalized := unfinalizedLowInversionMatcher(rows)
	verifyLowInversionLongBucket(t, unfinalized.longMatchBuckets.get(prefix), false)

	model, err := TrainModel(rows, WithThreshold(1))
	if err != nil {
		t.Fatalf("TrainModel: %v", err)
	}
	verifyLowInversionLongBucket(t, model.matcher.longMatchBuckets.get(prefix), true)

	suffixes := lowInversionSuffixes()
	probes := []string{
		string(lowInversionLongBucketPrefix[:]) + suffixes[0],
		string(lowInversionLongBucketPrefix[:]) + suffixes[lowInversionLeadingLongSuffix],
		string(lowInversionLongBucketPrefix[:]) + suffixes[lowInversionLeadingLongSuffix+1],
		suffixes[len(suffixes)-1],
	}
	archive, err := model.Encode(probes)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	for i, want := range probes {
		got, err := archive.AppendRow(nil, i)
		if err != nil {
			t.Fatalf("AppendRow(%d): %v", i, err)
		}
		if string(got) != want {
			t.Fatalf("round trip %d: got %q, want %q", i, got, want)
		}
	}
}

func BenchmarkLongBucketModelTrainLowInversion(b *testing.B) {
	rows := lowInversionLongBucketRows()
	inputBytes := 0
	for _, row := range rows {
		inputBytes += len(row)
	}
	prefix := bytesToU64LE(lowInversionLongBucketPrefix[:], minMatch)

	b.ReportAllocs()
	b.SetBytes(int64(inputBytes))
	var model *Model
	for b.Loop() {
		var err error
		model, err = TrainModel(rows, WithThreshold(1))
		if err != nil {
			b.Fatalf("TrainModel: %v", err)
		}
	}
	verifyLowInversionLongBucket(b, model.matcher.longMatchBuckets.get(prefix), true)
}
