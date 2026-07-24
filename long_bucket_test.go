package onpair

import (
	"slices"
	"testing"
)

const (
	longBucketPrefix          = "abcdefgh"
	longBucketBenchmarkLength = 2048
	longBucketEncodeRows      = 128
)

var longBucketPrefixKey = bytesToU64LE([]byte(longBucketPrefix), minMatch)

// progressiveLongPrefixRows creates a single row whose threshold-one training
// produces every prefix from two bytes through length. Once the prefix passes
// minMatch, every new token enters the same long-token bucket with a longer
// suffix than the preceding entry.
func progressiveLongPrefixRows(length int) []string {
	if length <= minMatch {
		panic("long bucket row must exceed minMatch")
	}

	row := make([]byte, length)
	copy(row, longBucketPrefix)
	for i := minMatch; i < len(row); i++ {
		row[i] = 'z'
	}
	return []string{string(row)}
}

func longBucketEntryCount(m *Model) int {
	bucket := m.matcher.longMatchBuckets.get(longBucketPrefixKey)
	if bucket == nil {
		return 0
	}
	return bucket.len()
}

func expectedProgressiveLongPrefixModel(row []byte) ([]byte, []uint32) {
	dictionary := make([]byte, singleByteTokens)
	boundaries := make([]uint32, singleByteTokens+1)
	for i := range dictionary {
		dictionary[i] = byte(i)
		boundaries[i] = uint32(i)
	}
	boundaries[singleByteTokens] = singleByteTokens

	for length := 2; length <= len(row); length++ {
		dictionary = append(dictionary, row[:length]...)
		boundaries = append(boundaries, uint32(len(dictionary)))
	}
	return dictionary, boundaries
}

func TestLongBucketTrainingPreservesGreedyMatches(t *testing.T) {
	rows := progressiveLongPrefixRows(longBucketBenchmarkLength)
	row := []byte(rows[0])

	model, err := TrainModel(rows, WithThreshold(1))
	if err != nil {
		t.Fatalf("TrainModel: %v", err)
	}

	wantEntries := len(row) - minMatch
	if got := longBucketEntryCount(model); got != wantEntries {
		t.Fatalf("long bucket entries: got %d, want %d", got, wantEntries)
	}
	t.Logf("long bucket entries: %d", wantEntries)

	wantDictionary, wantBoundaries := expectedProgressiveLongPrefixModel(row)
	if !slices.Equal(model.dictionary, wantDictionary) {
		t.Fatal("trained dictionary differs from the deterministic progressive-prefix dictionary")
	}
	if !slices.Equal(model.tokenBoundaries, wantBoundaries) {
		t.Fatal("trained token boundaries differ from the deterministic progressive-prefix boundaries")
	}

	checkMatch := func(length int) {
		probe := make([]byte, length+1)
		copy(probe, row[:length])
		probe[length] = '!'

		id, gotLength, ok := model.matcher.find(probe)
		wantID := uint16(singleByteTokens + length - 2)
		if !ok || id != wantID || gotLength != length {
			t.Fatalf("find(%d-byte token): got id=%d length=%d ok=%t, want id=%d length=%d", length, id, gotLength, ok, wantID, length)
		}
	}
	for length := 2; length < minMatch; length++ {
		checkMatch(length)
	}
	for length := minMatch + 1; length <= len(row); length++ {
		checkMatch(length)
	}

	queryRows := []string{
		rows[0],
		rows[0][:minMatch+1],
		rows[0][:minMatch+17] + "!",
		"unrelated",
	}
	archive, err := model.Encode(queryRows)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	for i, want := range queryRows {
		got, err := archive.AppendRow(nil, i)
		if err != nil {
			t.Fatalf("AppendRow(%d): %v", i, err)
		}
		if string(got) != want {
			t.Fatalf("row %d round trip: got %q, want %q", i, got, want)
		}
	}
}

// BenchmarkLongBucketModelTrain measures the public training boundary when a
// common eight-byte prefix accumulates progressively longer tokens. The model
// result is checked in the loop so the benchmark cannot discard training.
func BenchmarkLongBucketModelTrain(b *testing.B) {
	rows := progressiveLongPrefixRows(longBucketBenchmarkLength)
	wantEntries := len(rows[0]) - minMatch

	b.ReportAllocs()
	b.SetBytes(int64(len(rows[0])))
	for b.Loop() {
		model, err := TrainModel(rows, WithThreshold(1))
		if err != nil {
			b.Fatalf("TrainModel: %v", err)
		}
		if got := longBucketEntryCount(model); got != wantEntries {
			b.Fatalf("long bucket entries: got %d, want %d", got, wantEntries)
		}
	}
}

// BenchmarkLongBucketModelEncodeShortestHit guards the lookup case changed by
// an insertion-order index: a short matching token must scan a populated long
// bucket without regressing the greedy parser.
func BenchmarkLongBucketModelEncodeShortestHit(b *testing.B) {
	trainingRows := progressiveLongPrefixRows(longBucketBenchmarkLength)
	model, err := TrainModel(trainingRows, WithThreshold(1))
	if err != nil {
		b.Fatalf("TrainModel: %v", err)
	}

	shortestLongToken := trainingRows[0][:minMatch+1]
	queryRows := make([]string, longBucketEncodeRows)
	for i := range queryRows {
		queryRows[i] = shortestLongToken
	}

	b.ReportAllocs()
	b.SetBytes(int64(len(shortestLongToken) * len(queryRows)))
	for b.Loop() {
		archive, err := model.Encode(queryRows)
		if err != nil {
			b.Fatalf("Encode: %v", err)
		}
		if got := len(archive.CompressedData); got != len(queryRows) {
			b.Fatalf("compressed token count: got %d, want %d", got, len(queryRows))
		}
	}
}
