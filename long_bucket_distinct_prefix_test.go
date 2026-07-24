package onpair

import "testing"

const longBucketDistinctPrefixEntries = 2040

// distinctLongBucketRows makes one 9-byte long token per row. The first eight
// bytes are distinct, so threshold-one training places the fixed entry total
// into singleton long buckets instead of the one bucket in the primary fixture.
func distinctLongBucketRows(count int) []string {
	rows := make([]string, count)
	for i := range rows {
		row := [minMatch + 1]byte{
			byte(i), byte(i >> 8), 0x93, 0x4d, 0xe1, 0x27, 0xb8, 0x5a, 'z',
		}
		rows[i] = string(row[:])
	}
	return rows
}

// BenchmarkLongBucketModelTrainDistinctPrefixes guards the per-bucket
// finalization path. The primary training benchmark covers the matching
// one-bucket endpoint; this keeps the same 2,040 long-token total while
// exercising 2,040 distinct eight-byte-prefix buckets.
func BenchmarkLongBucketModelTrainDistinctPrefixes(b *testing.B) {
	rows := distinctLongBucketRows(longBucketDistinctPrefixEntries)

	b.ReportAllocs()
	b.SetBytes(int64(len(rows) * (minMatch + 1)))
	for b.Loop() {
		model, err := TrainModel(rows, WithThreshold(1))
		if err != nil {
			b.Fatalf("TrainModel: %v", err)
		}
		if got := model.matcher.longMatchBuckets.count; got != longBucketDistinctPrefixEntries {
			b.Fatalf("long bucket count: got %d, want %d", got, longBucketDistinctPrefixEntries)
		}
	}
}
