package onpair

import (
	"math/rand"
	"testing"
)

// randomizedSharedPrefixRows holds the corpus fixed while varying only the
// caller-visible row order. Encoder.train performs its own sampling shuffle,
// so this is an ordinary input-order sweep rather than a reconstruction of
// that implementation detail.
func randomizedSharedPrefixRows(rows []string) []string {
	shuffled := append([]string(nil), rows...)
	rand.New(rand.NewSource(1)).Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	return shuffled
}

func BenchmarkModelTrainSharedPrefixRandomizedInputLargeBucket(b *testing.B) {
	benchmarkModelTrainSharedPrefixRandomizedInputBucket(b, 256, 275)
}

func BenchmarkModelTrainSharedPrefixRandomizedInputExpandedBucket(b *testing.B) {
	// 8,192 variants occupy 720,896 B, below the default 1 MiB training sample
	// cap, and make an 8,622-entry primary-prefix bucket with this input order.
	benchmarkModelTrainSharedPrefixRandomizedInputBucket(b, 8192, 8622)
}

func benchmarkModelTrainSharedPrefixRandomizedInputBucket(b *testing.B, variants, wantBucketLen int) {
	rows := randomizedSharedPrefixRows(sharedPrefixMatchRows(variants))
	var totalBytes int64
	for _, row := range rows {
		totalBytes += int64(len(row))
	}

	b.ReportAllocs()
	b.SetBytes(totalBytes)
	b.ResetTimer()
	for b.Loop() {
		model := NewModel(WithThreshold(2))
		if err := model.Train(rows); err != nil {
			b.Fatalf("Model.Train: %v", err)
		}
		if got := sharedPrefixMatchBucketLen(model); got != wantBucketLen {
			b.Fatalf("shared-prefix bucket size: got %d want %d", got, wantBucketLen)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*int(totalBytes)), "ns/byte")
}
