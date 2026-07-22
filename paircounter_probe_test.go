//go:build perfprobe

package onpair

import "testing"

func TestPairCounterProbeHistogram(t *testing.T) {
	mobyDickRows, err := loadTestDataLines("testdata/en_mobydick.txt")
	if err != nil {
		t.Fatal(err)
	}
	mobyDick := trainAndCapturePairCounterProbe(t, "mobydick", mobyDickRows)
	if mobyDick.calls < 100_000 {
		t.Fatalf("Moby-Dick probe calls = %d, want at least 100000", mobyDick.calls)
	}

	highDistinctRows := highDistinctPairRows()
	highDistinct := trainAndCapturePairCounterProbe(
		t,
		"high-distinct",
		highDistinctRows,
		WithThreshold(65535),
		WithTrainingSampleBytes(len(highDistinctRows[0])),
	)
	if highDistinct.calls != highDistinctPairBytes-1 {
		t.Fatalf("high-distinct probe calls = %d, want %d", highDistinct.calls, highDistinctPairBytes-1)
	}
	if highDistinct.peakCapacity == 0 {
		t.Fatal("high-distinct probe did not record table capacity")
	}
	if load := float64(highDistinct.peakUsed) / float64(highDistinct.peakCapacity); load < 0.40 {
		t.Fatalf("high-distinct peak load = %.1f%%, want at least 40%%", load*100)
	}
	if p99 := pairCounterProbePercentile(highDistinct, 99, 100); p99 < 2 {
		t.Fatalf("high-distinct p99 probe length = %d, want at least 2", p99)
	}
}

func trainAndCapturePairCounterProbe(t *testing.T, name string, rows []string, opts ...Option) pairCounterProbeStats {
	t.Helper()
	resetPairCounterProbe()
	model := NewModel(opts...)
	if err := model.Train(rows); err != nil {
		t.Fatalf("train %s: %v", name, err)
	}
	if !model.Trained() {
		t.Fatalf("model was not trained for %s", name)
	}
	stats := snapshotPairCounterProbe()
	if stats.calls == 0 {
		t.Fatalf("no pair-counter probes recorded for %s", name)
	}
	logPairCounterProbe(t, name, stats)
	return stats
}

func logPairCounterProbe(t *testing.T, name string, stats pairCounterProbeStats) {
	t.Helper()
	average := float64(stats.totalSteps) / float64(stats.calls)
	t.Logf(
		"paircounter probe %s: calls=%d avg_steps=%.2f p95_steps=%d p99_steps=%d max_steps=%d peak_load=%.1f%%",
		name,
		stats.calls,
		average,
		pairCounterProbePercentile(stats, 95, 100),
		pairCounterProbePercentile(stats, 99, 100),
		stats.maxSteps,
		100*float64(stats.peakUsed)/float64(stats.peakCapacity),
	)
}

func pairCounterProbePercentile(stats pairCounterProbeStats, numerator, denominator uint64) int {
	if stats.calls == 0 {
		return 0
	}
	rank := (stats.calls*numerator + denominator - 1) / denominator
	var seen uint64
	for steps, calls := range stats.histogram {
		seen += calls
		if seen >= rank {
			return steps
		}
	}
	return stats.maxSteps
}
