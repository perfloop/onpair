package onpair

import "testing"

const highDistinctPairBytes = 13 * 1024

func highDistinctPairRows() []string {
	data := make([]byte, highDistinctPairBytes)
	state := uint32(0x6d2b79f5)
	for i := range data {
		state ^= state << 13
		state ^= state >> 17
		state ^= state << 5
		data[i] = byte(state >> 24)
	}
	return []string{string(data)}
}

func TestPairCounterCountsAcrossCollisionsTombstonesAndGrowth(t *testing.T) {
	counter := newPairCounter(1)
	initialCapacity := len(counter.entries)
	keys := makeCollidingPairCounterKeys(counter.mask, 96)

	for _, key := range keys {
		if got := counter.incr(key); got != 1 {
			t.Fatalf("first increment for %08x = %d, want 1", key, got)
		}
	}
	if len(counter.entries) <= initialCapacity {
		t.Fatalf("counter did not grow: capacity stayed at %d", initialCapacity)
	}

	for _, key := range keys {
		if got := counter.incr(key); got != 2 {
			t.Fatalf("second increment for %08x = %d, want 2", key, got)
		}
	}
	for i, key := range keys {
		if i%2 == 0 {
			counter.remove(key)
		}
	}
	for i, key := range keys {
		want := uint16(3)
		if i%2 == 0 {
			want = 1
		}
		if got := counter.incr(key); got != want {
			t.Fatalf("increment after tombstones for %08x = %d, want %d", key, got, want)
		}
	}
	for i, key := range keys {
		if i%2 == 0 {
			if got := counter.incr(key); got != 2 {
				t.Fatalf("second post-delete increment for %08x = %d, want 2", key, got)
			}
		}
	}
	if counter.count != len(keys) {
		t.Fatalf("live count = %d, want %d", counter.count, len(keys))
	}
}

func makeCollidingPairCounterKeys(mask uint32, count int) []uint32 {
	keys := make([]uint32, 0, count)
	slot := hashPairKey(0) & mask
	for key := uint32(0); len(keys) < count; key++ {
		if hashPairKey(key)&mask == slot {
			keys = append(keys, key)
		}
	}
	return keys
}

func BenchmarkPairCounterHighDistinctTrain(b *testing.B) {
	rows := highDistinctPairRows()
	sampleBytes := len(rows[0])

	b.ReportAllocs()
	b.SetBytes(int64(sampleBytes))
	for b.Loop() {
		model := NewModel(
			WithThreshold(65535),
			WithTrainingSampleBytes(sampleBytes),
		)
		if err := model.Train(rows); err != nil {
			b.Fatal(err)
		}
		if !model.Trained() || len(model.dictionary) != singleByteTokens {
			b.Fatalf("unexpected trained model: trained=%t dictionary bytes=%d", model.Trained(), len(model.dictionary))
		}
	}
}
