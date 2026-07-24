package onpair

import "testing"

type sharedSuffixHeadRejectFixture struct {
	model         *Model
	rows          []string
	totalBytes    int64
	expectedCodes int
}

func newSharedSuffixHeadRejectFixture(tb testing.TB, variants int) sharedSuffixHeadRejectFixture {
	tb.Helper()
	trainingRows := reverseTrainingDiscoveryOrder(sharedSuffixHeadRows(variants))
	rowLen := len(trainingRows[0])
	for i, row := range trainingRows {
		if len(row) != rowLen || row[:minMatch] != sharedSuffixHeadPrefix || row[minMatch:2*minMatch] != sharedSuffixHead {
			tb.Fatalf("training row %d does not retain the shared equal-length suffix head", i)
		}
	}

	model, err := TrainModel(trainingRows, WithThreshold(2))
	if err != nil {
		tb.Fatalf("TrainModel: %v", err)
	}
	bucket := model.matcher.longMatchBuckets.get(bytesToU64LE([]byte(sharedSuffixHeadPrefix), minMatch))
	if bucket == nil || bucket.len() < variants {
		got := 0
		if bucket != nil {
			got = bucket.len()
		}
		tb.Fatalf("trained shared-head bucket: got %d entries, want at least %d", got, variants)
	}

	// ffff is outside the generated 0..variants-1 tail range. It retains the
	// exact prefix, first suffix head, and row length while rejecting every full
	// long token in the trained group.
	reject := sharedSuffixHeadPrefix + sharedSuffixHead + "ffff" + adversarialLongBucketTail
	if len(reject) != rowLen {
		tb.Fatalf("reject length %d, want %d", len(reject), rowLen)
	}
	rows := make([]string, len(trainingRows))
	for i := range rows {
		rows[i] = reject
	}
	archive, err := model.Encode(rows)
	if err != nil {
		tb.Fatalf("Model.Encode setup: %v", err)
	}
	for _, index := range []int{0, len(rows) / 2, len(rows) - 1} {
		got, err := archive.AppendRow(nil, index)
		if err != nil {
			tb.Fatalf("AppendRow(%d): %v", index, err)
		}
		if string(got) != reject {
			tb.Fatalf("reject row %d round trip", index)
		}
	}

	return sharedSuffixHeadRejectFixture{
		model:         model,
		rows:          rows,
		totalBytes:    int64(len(reject) * len(rows)),
		expectedCodes: len(archive.CompressedData),
	}
}

func benchmarkModelEncodeSharedSuffixHeadReject(b *testing.B, variants int) {
	fixture := newSharedSuffixHeadRejectFixture(b, variants)
	b.ReportAllocs()
	b.SetBytes(fixture.totalBytes)
	b.ResetTimer()
	for b.Loop() {
		archive, err := fixture.model.Encode(fixture.rows)
		if err != nil {
			b.Fatalf("Model.Encode: %v", err)
		}
		if got := len(archive.CompressedData); got != fixture.expectedCodes {
			b.Fatalf("compressed token count: got %d want %d", got, fixture.expectedCodes)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*int(fixture.totalBytes)), "ns/byte")
}

func BenchmarkModelEncodeSharedSuffixHeadRejectLarge(b *testing.B) {
	benchmarkModelEncodeSharedSuffixHeadReject(b, 256)
}

func BenchmarkModelEncodeSharedSuffixHeadRejectExpanded(b *testing.B) {
	benchmarkModelEncodeSharedSuffixHeadReject(b, 8192)
}
