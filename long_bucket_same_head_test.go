package onpair

import (
	"fmt"
	"testing"
)

const (
	sharedSuffixHeadPrefix  = "abcdefgh"
	sharedSuffixHead        = "ABCDEFGH"
	sharedSuffixHeadRepeats = 4
)

func sharedSuffixHeadRows(variants int) []string {
	rows := make([]string, 0, variants*sharedSuffixHeadRepeats)
	for i := 0; i < variants; i++ {
		row := fmt.Sprintf("%s%s%04xqrstuvwxyz", sharedSuffixHeadPrefix, sharedSuffixHead, i)
		for range sharedSuffixHeadRepeats {
			rows = append(rows, row)
		}
	}
	return rows
}

// reverseTrainingDiscoveryOrder maps a caller-visible row order through the
// deterministic training shuffle so buildTokens sees the generated variants in
// reverse order. Every generated long token still has the same 8-byte suffix
// head and length; this only exercises the input-order boundary under which the
// former global ordered slice shifted candidates.
func reverseTrainingDiscoveryOrder(rows []string) []string {
	desired := append([]string(nil), rows...)
	for i, j := 0, len(desired)-1; i < j; i, j = i+1, j-1 {
		desired[i], desired[j] = desired[j], desired[i]
	}

	indices := make([]int, len(rows))
	for i := range indices {
		indices[i] = i
	}
	state := uint64(42)
	for i := len(indices) - 1; i > 0; i-- {
		state = state*6364136223846793005 + 1442695040888963407
		j := int(state % uint64(i+1))
		indices[i], indices[j] = indices[j], indices[i]
	}

	ordered := make([]string, len(rows))
	for i, rowIndex := range indices {
		ordered[rowIndex] = desired[i]
	}
	return ordered
}

func TestModelTrainSharedSuffixHeadReverseOrder(t *testing.T) {
	rows := reverseTrainingDiscoveryOrder(sharedSuffixHeadRows(256))
	model, err := TrainModel(rows, WithThreshold(2))
	if err != nil {
		t.Fatalf("TrainModel: %v", err)
	}
	archive, err := model.Encode(rows)
	if err != nil {
		t.Fatalf("Model.Encode: %v", err)
	}
	for _, index := range []int{0, len(rows) / 2, len(rows) - 1} {
		got, err := archive.AppendRow(nil, index)
		if err != nil {
			t.Fatalf("AppendRow(%d): %v", index, err)
		}
		if string(got) != rows[index] {
			t.Fatalf("row %d: got %q want %q", index, got, rows[index])
		}
	}
}

func BenchmarkModelTrainSharedSuffixHeadReverseLargeBucket(b *testing.B) {
	benchmarkModelTrainSharedSuffixHeadReverseBucket(b, 256)
}

func BenchmarkModelTrainSharedSuffixHeadReverseExpandedBucket(b *testing.B) {
	benchmarkModelTrainSharedSuffixHeadReverseBucket(b, 8192)
}

func benchmarkModelTrainSharedSuffixHeadReverseBucket(b *testing.B, variants int) {
	rows := reverseTrainingDiscoveryOrder(sharedSuffixHeadRows(variants))
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
		if model == nil {
			b.Fatal("Model.Train returned nil model")
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*int(totalBytes)), "ns/byte")
}
