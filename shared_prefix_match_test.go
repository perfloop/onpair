package onpair

import (
	"fmt"
	"testing"
)

const (
	sharedPrefixMatchPrefix  = "abcdefgh"
	sharedPrefixMatchRepeats = 4
)

type sharedPrefixMatchFixture struct {
	model      *Model
	rows       []string
	totalBytes int64
}

func sharedPrefixMatchRows(variants int) []string {
	rows := make([]string, 0, variants*sharedPrefixMatchRepeats)
	for i := 0; i < variants; i++ {
		row := fmt.Sprintf("%s%04xqrstuvwxyz", sharedPrefixMatchPrefix, i)
		for range sharedPrefixMatchRepeats {
			rows = append(rows, row)
		}
	}
	return rows
}

func sharedPrefixMatchBucketLen(model *Model) int {
	if model == nil || model.matcher == nil {
		return 0
	}
	key := bytesToU64LE([]byte(sharedPrefixMatchPrefix), minMatch)
	bucket := model.matcher.longMatchBuckets.get(key)
	if bucket == nil {
		return 0
	}
	return bucket.len()
}

func newSharedPrefixMatchFixture(tb testing.TB, variants, wantBucketLen int) sharedPrefixMatchFixture {
	tb.Helper()

	rows := sharedPrefixMatchRows(variants)
	model, err := TrainModel(rows, WithThreshold(2))
	if err != nil {
		tb.Fatalf("TrainModel: %v", err)
	}
	if got := sharedPrefixMatchBucketLen(model); got != wantBucketLen {
		tb.Fatalf("shared-prefix bucket size: got %d want %d", got, wantBucketLen)
	}

	var totalBytes int64
	for _, row := range rows {
		totalBytes += int64(len(row))
	}
	return sharedPrefixMatchFixture{model: model, rows: rows, totalBytes: totalBytes}
}

func TestMatcherFindSharedPrefixTailAndGreedyOrder(t *testing.T) {
	matcher := newMatcher(0)
	tokens := []string{
		"abcdefghABC",
		"abcdefghABCDEFGH",
		"abcdefghABCDEFGHtail",
		"abcdefghABCDEFGHtale",
		"abcdefghABCDEFGHtail-extra",
	}
	for i, token := range tokens {
		if !matcher.insert([]byte(token), uint16(singleByteTokens+i)) {
			t.Fatalf("insert %q", token)
		}
	}

	cases := []struct {
		input  string
		id     uint16
		length int
	}{
		{"abcdefghABCDEFGHtail-extra/next", singleByteTokens + 4, len(tokens[4])},
		{"abcdefghABCDEFGHtale/next", singleByteTokens + 3, len(tokens[3])},
		{"abcdefghABCDEFGHtail?", singleByteTokens + 2, len(tokens[2])},
		{"abcdefghABC!", singleByteTokens, len(tokens[0])},
	}
	for _, tc := range cases {
		id, length, ok := matcher.find([]byte(tc.input))
		if !ok {
			t.Fatalf("find(%q) did not match", tc.input)
		}
		if id != uint16(tc.id) || length != tc.length {
			t.Fatalf("find(%q): got id=%d length=%d want id=%d length=%d", tc.input, id, length, tc.id, tc.length)
		}
	}
}

func TestModelEncodeSharedPrefixBucketRoundTrip(t *testing.T) {
	fixture := newSharedPrefixMatchFixture(t, 256, 273)
	archive, err := fixture.model.Encode(fixture.rows)
	if err != nil {
		t.Fatalf("Model.Encode: %v", err)
	}
	if got, want := len(archive.CompressedData), len(fixture.rows); got != want {
		t.Fatalf("compressed token count: got %d want %d", got, want)
	}

	for i, want := range fixture.rows {
		got, err := archive.AppendRow(nil, i)
		if err != nil {
			t.Fatalf("AppendRow(%d): %v", i, err)
		}
		if string(got) != want {
			t.Fatalf("row %d: got %q want %q", i, got, want)
		}
	}
}

func BenchmarkModelEncodeSharedPrefixLargeBucket(b *testing.B) {
	benchmarkModelEncodeSharedPrefixBucket(b, 256, 273)
}

func BenchmarkModelEncodeSharedPrefixSmallBucket(b *testing.B) {
	benchmarkModelEncodeSharedPrefixBucket(b, 64, 71)
}

func benchmarkModelEncodeSharedPrefixBucket(b *testing.B, variants, wantBucketLen int) {
	fixture := newSharedPrefixMatchFixture(b, variants, wantBucketLen)
	alternateRows := append([]string(nil), fixture.rows...)
	for i, j := 0, len(alternateRows)-1; i < j; i, j = i+1, j-1 {
		alternateRows[i], alternateRows[j] = alternateRows[j], alternateRows[i]
	}

	for _, rows := range [][]string{fixture.rows, alternateRows} {
		archive, err := fixture.model.Encode(rows)
		if err != nil {
			b.Fatalf("Model.Encode setup: %v", err)
		}
		if got, want := len(archive.CompressedData), len(rows); got != want {
			b.Fatalf("Model.Encode setup tokens: got %d want %d", got, want)
		}
	}

	b.ReportAllocs()
	b.SetBytes(fixture.totalBytes)
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		rows := fixture.rows
		if i&1 != 0 {
			rows = alternateRows
		}
		archive, err := fixture.model.Encode(rows)
		if err != nil {
			b.Fatalf("Model.Encode: %v", err)
		}
		if got, want := len(archive.CompressedData), len(rows); got != want {
			b.Fatalf("compressed token count: got %d want %d", got, want)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*int(fixture.totalBytes)), "ns/byte")
}

func BenchmarkModelTrainSharedPrefixLargeBucket(b *testing.B) {
	fixture := newSharedPrefixMatchFixture(b, 256, 273)

	b.ReportAllocs()
	b.SetBytes(fixture.totalBytes)
	b.ResetTimer()
	for b.Loop() {
		model := NewModel(WithThreshold(2))
		if err := model.Train(fixture.rows); err != nil {
			b.Fatalf("Model.Train: %v", err)
		}
		if got := sharedPrefixMatchBucketLen(model); got != 273 {
			b.Fatalf("shared-prefix bucket size: got %d want %d", got, 273)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*int(fixture.totalBytes)), "ns/byte")
}
