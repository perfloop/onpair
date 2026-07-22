package onpair

import (
	"runtime"
	"slices"
	"testing"
)

const sampledTrainingBenchmarkRows = 1 << 20

func TestModelTrainSampledDeterministic(t *testing.T) {
	rows := makeSyntheticMixedRows(1 << 16)
	data, _ := flattenStrings(rows)
	if len(data) <= maxTrainingSampleBytes {
		t.Fatalf("test input must exceed the default sample budget: got %d bytes", len(data))
	}

	tests := []struct {
		name string
		opts []Option
	}{
		{name: "plain"},
		{
			name: "template_stratified",
			opts: []Option{WithTemplateStratifiedSampling(32)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			first, err := TrainModel(rows, tt.opts...)
			if err != nil {
				t.Fatalf("first TrainModel: %v", err)
			}
			second, err := TrainModel(rows, tt.opts...)
			if err != nil {
				t.Fatalf("second TrainModel: %v", err)
			}
			if !slices.Equal(first.dictionary, second.dictionary) {
				t.Fatal("sampled training produced a non-deterministic dictionary")
			}
			if !slices.Equal(first.tokenBoundaries, second.tokenBoundaries) {
				t.Fatal("sampled training produced non-deterministic token boundaries")
			}

			archive, err := first.Encode(rows[:64])
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
			verifyArchiveRoundTrip(t, archive, rows[:64])
		})
	}
}

func BenchmarkEncoderTrainSampledLargeBatch(b *testing.B) {
	rows := makeSyntheticMixedRows(sampledTrainingBenchmarkRows)
	data, endPositions := flattenStrings(rows)
	rows = nil
	if len(data) <= maxTrainingSampleBytes {
		b.Fatalf("benchmark input must exceed the default sample budget: got %d bytes", len(data))
	}

	// Keep the timed operation to Encoder.train: Model.Train's flattening cost is
	// intentionally outside this benchmark, while this input leaves a large tail
	// of rows after the default 1 MiB sample is filled.
	runtime.GC()
	encoder := NewEncoder()
	var (
		gotMatcher      *matcher
		dictionary      []byte
		tokenBoundaries []uint32
	)
	b.ReportAllocs()
	for b.Loop() {
		gotMatcher, dictionary, tokenBoundaries = encoder.train(data, endPositions)
	}
	if gotMatcher == nil || len(tokenBoundaries) == 0 || len(dictionary) < singleByteTokens {
		b.Fatal("train returned an incomplete dictionary")
	}
	if got := int(tokenBoundaries[len(tokenBoundaries)-1]); got != len(dictionary) {
		b.Fatalf("last token boundary = %d, dictionary length = %d", got, len(dictionary))
	}
}
