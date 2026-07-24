package onpair

import (
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"testing"
)

func TestModelTrainEncode(t *testing.T) {
	input := []string{
		"user_000001",
		"user_000002",
		"admin_001",
	}

	model, err := TrainModel(input, WithMaxTokenLength(16))
	if err != nil {
		t.Fatalf("TrainModel failed: %v", err)
	}
	if !model.Trained() {
		t.Fatalf("model should be trained")
	}

	archive, err := model.Encode(input)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	buf := make([]byte, 256)
	for i, want := range input {
		n, err := archive.DecompressString(i, buf)
		if err != nil {
			t.Fatalf("DecompressString(%d) failed: %v", i, err)
		}
		if got := string(buf[:n]); got != want {
			t.Fatalf("row %d mismatch: got %q want %q", i, got, want)
		}
	}
}

func TestModelOnPair16SharedPrefixGreedyRoundTrip(t *testing.T) {
	baseRows := []string{
		"abcdefghA",
		"abcdefghABC",
		"abcdefghABCD",
		"abcdefghABCE",
		"abcdefghABCDEFGH",
		"abcdefghZ",
	}
	trainingRows := make([]string, 0, len(baseRows)*16)
	for _, row := range baseRows {
		for range 16 {
			trainingRows = append(trainingRows, row)
		}
	}

	model, err := TrainModel(trainingRows, WithThreshold(2), WithMaxTokenLength(16))
	if err != nil {
		t.Fatalf("TrainModel: %v", err)
	}
	bucket := model.matcher.longMatchBuckets.get(bytesToU64LE([]byte("abcdefgh"), minMatch))
	if bucket == nil || bucket.len() < len(baseRows) {
		got := 0
		if bucket != nil {
			got = bucket.len()
		}
		t.Fatalf("trained OnPair16 shared-prefix bucket: got %d entries, want at least %d", got, len(baseRows))
	}

	probes := []struct {
		row        string
		wantPrefix string
	}{
		{"abcdefghABCDEFGH/next", "abcdefghABCDEFGH"},
		{"abcdefghABCE/next", "abcdefghABCE"},
		{"abcdefghABCD/next", "abcdefghABCD"},
		{"abcdefghABC/next", "abcdefghABC"},
		{"abcdefghABCF/next", "abcdefghABC"}, // reject longer ABCD/ABCE and fall back greedily
		{"abcdefghA/next", "abcdefghA"},
	}
	rows := make([]string, len(probes))
	for i, probe := range probes {
		rows[i] = probe.row
	}
	archive, err := model.Encode(rows)
	if err != nil {
		t.Fatalf("Model.Encode: %v", err)
	}
	for i, probe := range probes {
		got, err := archive.AppendRow(nil, i)
		if err != nil {
			t.Fatalf("AppendRow(%d): %v", i, err)
		}
		if string(got) != probe.row {
			t.Fatalf("round trip row %d: got %q want %q", i, got, probe.row)
		}
		start := archive.StringBoundaries[i]
		id := archive.CompressedData[start]
		first := model.tokenBoundaries[id]
		last := model.tokenBoundaries[id+1]
		if gotPrefix := string(model.dictionary[first:last]); gotPrefix != probe.wantPrefix {
			t.Fatalf("row %d first greedy token: got %q want %q", i, gotPrefix, probe.wantPrefix)
		}
	}
}

const (
	modelSharedKeyPrefix = "abcdefgh"
	modelSharedKeyHead   = "ABCDEFGHIJKLMNOP"
	modelSharedKeyTail   = "xy"
)

func modelSharedKeyRows(variants int) []string {
	rows := make([]string, 0, variants*4)
	for i := 0; i < variants; i++ {
		row := fmt.Sprintf("%s%s%04x%s", modelSharedKeyPrefix, modelSharedKeyHead, i, modelSharedKeyTail)
		rows = append(rows, row, row, row, row)
	}
	return rows
}

func reverseModelSharedKeyDiscovery(rows []string) []string {
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

func trainModelSharedKeyFixture(tb testing.TB, variants int, reverse bool, sampleBytes int) (*Model, []string) {
	tb.Helper()
	rows := modelSharedKeyRows(variants)
	if reverse {
		rows = reverseModelSharedKeyDiscovery(rows)
	}
	for i, row := range rows {
		if len(row) != 30 || row[:minMatch] != modelSharedKeyPrefix || row[minMatch:minMatch+len(modelSharedKeyHead)] != modelSharedKeyHead {
			tb.Fatalf("row %d does not retain the common 16-byte suffix key", i)
		}
	}
	opts := []Option{WithThreshold(2)}
	if sampleBytes != 0 {
		opts = append(opts, WithTrainingSampleBytes(sampleBytes))
	}
	model, err := TrainModel(rows, opts...)
	if err != nil {
		tb.Fatalf("TrainModel: %v", err)
	}
	bucket := model.matcher.longMatchBuckets.get(bytesToU64LE([]byte(modelSharedKeyPrefix), minMatch))
	got := 0
	if bucket != nil {
		got = bucket.len()
	}
	if got < variants {
		tb.Fatalf("trained shared-key bucket: got %d entries, want at least %d", got, variants)
	}
	return model, rows
}

func TestModelSharedSuffixKeyFixture(t *testing.T) {
	model, rows := trainModelSharedKeyFixture(t, 273, false, 0)
	reject := fmt.Sprintf("%s%sffff%s", modelSharedKeyPrefix, modelSharedKeyHead, modelSharedKeyTail)
	archive, err := model.Encode([]string{rows[0], rows[len(rows)/2], reject})
	if err != nil {
		t.Fatalf("Model.Encode: %v", err)
	}
	for i, want := range []string{rows[0], rows[len(rows)/2], reject} {
		got, err := archive.AppendRow(nil, i)
		if err != nil {
			t.Fatalf("AppendRow(%d): %v", i, err)
		}
		if string(got) != want {
			t.Fatalf("row %d round trip: got %q want %q", i, got, want)
		}
	}
	if _, matchLen, ok := model.matcher.find([]byte(reject)); !ok || matchLen >= len(reject) {
		t.Fatalf("rejecting probe should fall back from the full shared-key token: match length %d, found %t", matchLen, ok)
	}
}

func benchmarkModelTrainSharedSuffixKeyReverse(b *testing.B, variants, sampleBytes int) {
	rows := reverseModelSharedKeyDiscovery(modelSharedKeyRows(variants))
	totalBytes := int64(0)
	for _, row := range rows {
		totalBytes += int64(len(row))
	}
	opts := []Option{WithThreshold(2)}
	if sampleBytes != 0 {
		opts = append(opts, WithTrainingSampleBytes(sampleBytes))
	}
	b.ReportAllocs()
	b.SetBytes(totalBytes)
	b.ResetTimer()
	for b.Loop() {
		model := NewModel(opts...)
		if err := model.Train(rows); err != nil {
			b.Fatal(err)
		}
		if model == nil {
			b.Fatal("nil model")
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*int(totalBytes)), "ns/byte")
}

func benchmarkModelEncodeSharedSuffixKeyReject(b *testing.B, variants, sampleBytes int) {
	model, rows := trainModelSharedKeyFixture(b, variants, true, sampleBytes)
	reject := fmt.Sprintf("%s%sffff%s", modelSharedKeyPrefix, modelSharedKeyHead, modelSharedKeyTail)
	for i := range rows {
		rows[i] = reject
	}
	archive, err := model.Encode(rows)
	if err != nil {
		b.Fatal(err)
	}
	want := len(archive.CompressedData)
	totalBytes := int64(len(reject) * len(rows))
	b.ReportAllocs()
	b.SetBytes(totalBytes)
	b.ResetTimer()
	for b.Loop() {
		archive, err := model.Encode(rows)
		if err != nil {
			b.Fatal(err)
		}
		if len(archive.CompressedData) != want {
			b.Fatal("compressed token count changed")
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*int(totalBytes)), "ns/byte")
}

func TestModelEncodeWithoutTrain(t *testing.T) {
	model := NewModel()
	_, err := model.Encode([]string{"x"})
	if !errors.Is(err, ErrUntrainedModel) {
		t.Fatalf("expected ErrUntrainedModel, got %v", err)
	}
}

func FuzzModelLifecycle(f *testing.F) {
	f.Add([]byte("user_"), []byte("0001"), []byte("event"))
	f.Add([]byte(""), []byte(""), []byte(""))
	f.Add([]byte("prefix"), []byte("_suffix"), []byte("payload"))

	f.Fuzz(func(t *testing.T, trainA, trainB, query []byte) {
		total := len(trainA) + len(trainB) + len(query)
		if limitFuzzSize(total) {
			t.Skip()
		}

		trainRows := []string{
			string(trainA),
			string(trainB),
			string(trainA) + string(trainB),
			string(trainA),
		}
		queryRows := []string{
			string(query),
			string(trainA),
			string(query) + string(trainB),
			string(query),
		}

		model, err := TrainModel(trainRows, WithMaxTokenLength(16))
		if err != nil {
			t.Fatalf("TrainModel failed: %v", err)
		}
		if !model.Trained() {
			t.Fatalf("model should be trained")
		}

		archive1, err := model.Encode(queryRows)
		if err != nil {
			t.Fatalf("first model encode failed: %v", err)
		}
		archive2, err := model.Encode(queryRows)
		if err != nil {
			t.Fatalf("second model encode failed: %v", err)
		}

		if !slices.Equal(archive1.CompressedData, archive2.CompressedData) {
			t.Fatalf("non-deterministic compressed data")
		}
		if !slices.Equal(archive1.StringBoundaries, archive2.StringBoundaries) {
			t.Fatalf("non-deterministic string boundaries")
		}
		if !slices.Equal(archive1.Dictionary, archive2.Dictionary) {
			t.Fatalf("non-deterministic dictionary")
		}
		if !slices.Equal(archive1.TokenBoundaries, archive2.TokenBoundaries) {
			t.Fatalf("non-deterministic token boundaries")
		}

		verifyArchiveRoundTrip(t, archive1, queryRows)
	})
}

// benchTrainEncodeDatasets lists representative files for isolated train vs
// encode benchmarks. Keep small to limit bench runtime; pick datasets that
// exercise distinct token-length distributions.
var benchTrainEncodeDatasets = []string{
	"testdata/logs_apache_2k.log",
	"testdata/logs_hdfs_2k.log",
	"testdata/en_mobydick.txt",
	"testdata/en_shakespeare.txt",
}

func benchDatasetBytes(lines []string) int64 {
	n := 0
	for _, line := range lines {
		n += len(line)
	}
	return int64(n)
}

// BenchmarkModelTrain measures Model.Train() in isolation.
func BenchmarkModelTrain(b *testing.B) {
	for _, path := range benchTrainEncodeDatasets {
		lines, err := loadTestDataLines(path)
		if err != nil || len(lines) == 0 {
			b.Logf("skip %s: %v", path, err)
			continue
		}
		name := filepath.Base(path)
		total := benchDatasetBytes(lines)

		b.Run(name+"/OnPair", func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(total)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m := NewModel()
				if err := m.Train(lines); err != nil {
					b.Fatalf("train: %v", err)
				}
			}
		})

		b.Run(name+"/OnPair16", func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(total)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m := NewModel(WithMaxTokenLength(16))
				if err := m.Train(lines); err != nil {
					b.Fatalf("train: %v", err)
				}
			}
		})
	}
}

// BenchmarkModelEncode measures Model.Encode() in isolation using a pre-trained model.
func BenchmarkModelEncode(b *testing.B) {
	for _, path := range benchTrainEncodeDatasets {
		lines, err := loadTestDataLines(path)
		if err != nil || len(lines) == 0 {
			b.Logf("skip %s: %v", path, err)
			continue
		}
		name := filepath.Base(path)
		total := benchDatasetBytes(lines)

		model, err := TrainModel(lines)
		if err != nil {
			b.Fatalf("pre-train %s: %v", path, err)
		}
		model16, err := TrainModel(lines, WithMaxTokenLength(16))
		if err != nil {
			b.Fatalf("pre-train16 %s: %v", path, err)
		}

		b.Run(name+"/OnPair", func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(total)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := model.Encode(lines); err != nil {
					b.Fatalf("encode: %v", err)
				}
			}
		})

		b.Run(name+"/OnPair16", func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(total)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := model16.Encode(lines); err != nil {
					b.Fatalf("encode: %v", err)
				}
			}
		})
	}
}

func BenchmarkModelTrainSharedSuffixKeyReverseLarge(b *testing.B) {
	benchmarkModelTrainSharedSuffixKeyReverse(b, 273, 0)
}

func BenchmarkModelTrainSharedSuffixKeyReverseExpanded(b *testing.B) {
	benchmarkModelTrainSharedSuffixKeyReverse(b, 8192, 0)
}

func BenchmarkModelEncodeSharedSuffixKeyRejectLarge(b *testing.B) {
	benchmarkModelEncodeSharedSuffixKeyReject(b, 273, 0)
}

func BenchmarkModelEncodeSharedSuffixKeyRejectExpanded(b *testing.B) {
	benchmarkModelEncodeSharedSuffixKeyReject(b, 8192, 0)
}

func BenchmarkModelTrainSharedSuffixKeyReverseConfiguredExpanded(b *testing.B) {
	benchmarkModelTrainSharedSuffixKeyReverse(b, 16384, 2*1024*1024)
}

func BenchmarkModelEncodeSharedSuffixKeyRejectConfiguredExpanded(b *testing.B) {
	benchmarkModelEncodeSharedSuffixKeyReject(b, 16384, 2*1024*1024)
}
