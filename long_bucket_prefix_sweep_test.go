package onpair

import "testing"

const longBucketSweepEntries = 2040

func longBucketSweepPrefix(index int) [minMatch]byte {
	return [minMatch]byte{
		byte(index), byte(index >> 8), 0x91, 0x3e, 0xd4, 0x67, 0xb8, 0x2a,
	}
}

// distributedLongBucketRows keeps the long-token total fixed while splitting
// it over the requested number of distinct eight-byte-prefix buckets.
func distributedLongBucketRows(bucketCount int) ([]string, map[uint64]int) {
	if bucketCount < 1 || bucketCount > longBucketSweepEntries {
		panic("invalid long-bucket sweep count")
	}

	rows := make([]string, bucketCount)
	expected := make(map[uint64]int, bucketCount)
	baseEntries := longBucketSweepEntries / bucketCount
	extraEntries := longBucketSweepEntries % bucketCount
	for i := range rows {
		entryCount := baseEntries
		if i < extraEntries {
			entryCount++
		}
		prefix := longBucketSweepPrefix(i)
		row := make([]byte, minMatch+entryCount)
		copy(row, prefix[:])
		for j := minMatch; j < len(row); j++ {
			row[j] = 'z'
		}
		rows[i] = string(row)
		expected[bytesToU64LE(prefix[:], minMatch)] = entryCount
	}
	return rows, expected
}

func verifyDistributedLongBucketModel(tb testing.TB, model *Model, expected map[uint64]int) {
	tb.Helper()
	if got := model.matcher.longMatchBuckets.count; got != len(expected) {
		tb.Fatalf("long bucket count: got %d, want %d", got, len(expected))
	}

	entries := model.matcher.longMatchBuckets.entries
	total := 0
	for i := range entries {
		entry := entries[i]
		if entry.bucket == nil {
			continue
		}
		wantEntries, ok := expected[entry.key]
		if !ok {
			tb.Fatalf("unexpected long bucket %#x", entry.key)
		}
		if got := entry.bucket.len(); got != wantEntries {
			tb.Fatalf("bucket %#x entries: got %d, want %d", entry.key, got, wantEntries)
		}
		for j, got := range entry.bucket.suffixLens {
			want := uint16(wantEntries - j)
			if got != want {
				tb.Fatalf("bucket %#x suffix length %d: got %d, want %d", entry.key, j, got, want)
			}
		}
		total += entry.bucket.len()
	}
	if total != longBucketSweepEntries {
		tb.Fatalf("long entry total: got %d, want %d", total, longBucketSweepEntries)
	}
}

func benchmarkDistributedLongBucketTrain(b *testing.B, bucketCount int) {
	rows, expected := distributedLongBucketRows(bucketCount)
	inputBytes := 0
	for _, row := range rows {
		inputBytes += len(row)
	}

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
	verifyDistributedLongBucketModel(b, model, expected)
}

func BenchmarkLongBucketModelTrainPrefixBuckets8(b *testing.B) {
	benchmarkDistributedLongBucketTrain(b, 8)
}

func BenchmarkLongBucketModelTrainPrefixBuckets120(b *testing.B) {
	benchmarkDistributedLongBucketTrain(b, 120)
}

func BenchmarkLongBucketModelTrainPrefixBuckets1020(b *testing.B) {
	benchmarkDistributedLongBucketTrain(b, 1020)
}

const longBucketEqualSuffixEntries = 2040

var longBucketEqualSuffixPrefix = [minMatch]byte{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}

func deterministicTrainingOrder(n int) []int {
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	state := uint64(42)
	for i := len(order) - 1; i > 0; i-- {
		state = state*6364136223846793005 + 1442695040888963407
		j := int(state % uint64(i+1))
		order[i], order[j] = order[j], order[i]
	}
	return order
}

// equalSuffixLongBucketRows arranges TrainModel's deterministic row order into
// three phases: build the shared short prefix, build each two-byte suffix, then
// merge that prefix and suffix. It therefore produces one 2-byte-suffix long
// token per row in a common bucket; all entries have equal suffix length, so
// the pre-image insertion loop performs no payload swaps.
func equalSuffixLongBucketRows(count int) []string {
	phases := make([]string, 0, 1+2*count)
	phases = append(phases, string(longBucketEqualSuffixPrefix[:]))
	for i := 0; i < count; i++ {
		suffix := [2]byte{byte(i), byte(i >> 8)}
		phases = append(phases, string(suffix[:]))
	}
	for i := 0; i < count; i++ {
		suffix := [2]byte{byte(i), byte(i >> 8)}
		row := [minMatch + 2]byte{
			longBucketEqualSuffixPrefix[0], longBucketEqualSuffixPrefix[1],
			longBucketEqualSuffixPrefix[2], longBucketEqualSuffixPrefix[3],
			longBucketEqualSuffixPrefix[4], longBucketEqualSuffixPrefix[5],
			longBucketEqualSuffixPrefix[6], longBucketEqualSuffixPrefix[7],
			suffix[0], suffix[1],
		}
		phases = append(phases, string(row[:]))
	}

	rows := make([]string, len(phases))
	for phase, index := range deterministicTrainingOrder(len(rows)) {
		rows[index] = phases[phase]
	}
	return rows
}

func verifyEqualSuffixLongBucketModel(tb testing.TB, model *Model, count int) {
	tb.Helper()
	prefix := bytesToU64LE(longBucketEqualSuffixPrefix[:], minMatch)
	bucket := model.matcher.longMatchBuckets.get(prefix)
	if bucket == nil {
		tb.Fatal("missing equal-suffix long bucket")
	}
	if got := model.matcher.longMatchBuckets.count; got != 1 {
		tb.Fatalf("long bucket count: got %d, want 1", got)
	}
	if got := bucket.len(); got != count {
		tb.Fatalf("equal-suffix bucket entries: got %d, want %d", got, count)
	}
	for i, suffixLen := range bucket.suffixLens {
		if suffixLen != 2 {
			tb.Fatalf("equal-suffix entry %d: got suffix length %d, want 2", i, suffixLen)
		}
	}
}

func TestLongBucketTrainingPreservesEqualSuffixFixture(t *testing.T) {
	model, err := TrainModel(equalSuffixLongBucketRows(longBucketEqualSuffixEntries), WithThreshold(1))
	if err != nil {
		t.Fatalf("TrainModel: %v", err)
	}
	verifyEqualSuffixLongBucketModel(t, model, longBucketEqualSuffixEntries)
}

func BenchmarkLongBucketModelTrainEqualSuffixes(b *testing.B) {
	rows := equalSuffixLongBucketRows(longBucketEqualSuffixEntries)
	inputBytes := 0
	for _, row := range rows {
		inputBytes += len(row)
	}

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
	verifyEqualSuffixLongBucketModel(b, model, longBucketEqualSuffixEntries)
}
