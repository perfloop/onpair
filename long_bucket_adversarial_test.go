package onpair

import (
	"encoding/binary"
	"math/bits"
	"sync"
	"testing"
)

const adversarialLongBucketTail = "qrstuvwxyz"

type adversarialHeadKey struct {
	slotCount int
	count     int
	colliding bool
}

var adversarialHeadsCache = struct {
	sync.Mutex
	values map[adversarialHeadKey][]uint64
}{values: make(map[adversarialHeadKey][]uint64)}

// adversarialGroupHash and adversarialGroupStartSlot reproduce the preimage's
// public deterministic group-slot calculation. The fixture uses them to make
// distinct suffix heads that begin in one collision cluster, rather than a
// same-head chain. The reported probe count is deliberately labeled preimage:
// it quantifies the selected attack input rather than assuming a keyed
// implementation retains those collisions.
func adversarialGroupHash(head uint64, headLen int) uint64 {
	head ^= uint64(headLen) * 0x9e3779b97f4a7c15
	head ^= head >> 30
	head *= 0xbf58476d1ce4e5b9
	head ^= head >> 27
	head *= 0x94d049bb133111eb
	return head ^ (head >> 31)
}

func adversarialGroupStartSlot(head uint64, slotCount int) int {
	high, _ := bits.Mul64(adversarialGroupHash(head, minMatch), uint64(slotCount))
	return int(high)
}

func adversarialSplitMix64(value uint64) uint64 {
	value += 0x9e3779b97f4a7c15
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	return value ^ (value >> 31)
}

// adversarialPrintableHead makes the collision search exercise the same
// ASCII-token training shape as the existing shared-prefix workloads. It
// packs eight hexadecimal bytes in little-endian order, which is exactly the
// representation matcher.find reads from the suffix head.
func adversarialPrintableHead(value uint64) uint64 {
	const hex = "0123456789abcdef"
	value = adversarialSplitMix64(value)
	var head uint64
	for shift := 0; shift < 64; shift += 8 {
		head |= uint64(hex[value&0xf]) << shift
		value >>= 4
	}
	return head
}

func adversarialHeads(slotCount, count int, colliding bool) []uint64 {
	key := adversarialHeadKey{slotCount: slotCount, count: count, colliding: colliding}
	adversarialHeadsCache.Lock()
	if heads := adversarialHeadsCache.values[key]; heads != nil {
		adversarialHeadsCache.Unlock()
		return heads
	}
	adversarialHeadsCache.Unlock()

	heads := make([]uint64, 0, count)
	seenHeads := make(map[uint64]struct{}, count)
	seenSlots := make(map[int]struct{}, count)
	for value := uint64(1); len(heads) < count; value++ {
		head := adversarialPrintableHead(value)
		slot := adversarialGroupStartSlot(head, slotCount)
		if colliding {
			if slot != 0 {
				continue
			}
		} else {
			if _, exists := seenSlots[slot]; exists {
				continue
			}
			seenSlots[slot] = struct{}{}
		}
		if _, exists := seenHeads[head]; exists {
			continue
		}
		seenHeads[head] = struct{}{}
		heads = append(heads, head)
	}

	adversarialHeadsCache.Lock()
	adversarialHeadsCache.values[key] = heads
	adversarialHeadsCache.Unlock()
	return heads
}

func adversarialToken(head uint64) []byte {
	token := make([]byte, 0, minMatch+minMatch+len(adversarialLongBucketTail))
	token = append(token, sharedSuffixHeadPrefix...)
	var packed [minMatch]byte
	binary.LittleEndian.PutUint64(packed[:], head)
	token = append(token, packed[:]...)
	token = append(token, adversarialLongBucketTail...)
	return token
}

func adversarialCollisionRows(variants, slotCount int, colliding bool) ([]string, []uint64) {
	heads := adversarialHeads(slotCount, variants+1, colliding)
	rows := make([]string, 0, variants*sharedSuffixHeadRepeats)
	for _, head := range heads[:variants] {
		row := string(adversarialToken(head))
		for range sharedSuffixHeadRepeats {
			rows = append(rows, row)
		}
	}
	return rows, heads
}

func adversarialSlotProbes(heads []uint64, slotCount int) int {
	slots := make([]bool, slotCount)
	probes := 0
	for _, head := range heads {
		slot := adversarialGroupStartSlot(head, slotCount)
		for {
			probes++
			if !slots[slot] {
				slots[slot] = true
				break
			}
			slot++
			if slot == len(slots) {
				slot = 0
			}
		}
	}
	return probes
}

func verifyAdversarialHeadShape(tb testing.TB, heads []uint64, slotCount int, colliding bool) {
	tb.Helper()
	if len(heads) < 2 {
		tb.Fatal("need a rejecting head after inserted heads")
	}
	firstSlot := adversarialGroupStartSlot(heads[0], slotCount)
	seenSlots := make(map[int]struct{}, len(heads))
	for i, head := range heads {
		slot := adversarialGroupStartSlot(head, slotCount)
		if colliding {
			if slot != firstSlot {
				tb.Fatalf("head %d starts at slot %d, want %d", i, slot, firstSlot)
			}
		} else {
			if _, exists := seenSlots[slot]; exists {
				tb.Fatalf("head %d repeats distributed slot %d", i, slot)
			}
			seenSlots[slot] = struct{}{}
		}
	}
}

func verifyAdversarialTrainFixture(tb testing.TB, rows []string, variants int) {
	tb.Helper()
	model, err := TrainModel(rows, WithThreshold(2))
	if err != nil {
		tb.Fatalf("TrainModel: %v", err)
	}
	bucket := model.matcher.longMatchBuckets.get(bytesToU64LE([]byte(sharedSuffixHeadPrefix), minMatch))
	if bucket == nil || bucket.len() < variants {
		got := 0
		if bucket != nil {
			got = bucket.len()
		}
		tb.Fatalf("trained shared-prefix bucket: got %d entries, want at least %d", got, variants)
	}
}

func TestLongBucketHashFloodFixtureProperties(t *testing.T) {
	for _, slotCount := range []int{256, 640, 1600, 4000, 10000} {
		verifyAdversarialHeadShape(t, adversarialHeads(slotCount, 8, true), slotCount, true)
		verifyAdversarialHeadShape(t, adversarialHeads(slotCount, 8, false), slotCount, false)
	}
}

func TestSharedSuffixHeadReverseFixtureProperties(t *testing.T) {
	for _, variants := range []int{256, 8192} {
		rows := reverseTrainingDiscoveryOrder(sharedSuffixHeadRows(variants))
		rowLen := len(rows[0])
		for i, row := range rows {
			if len(row) != rowLen || row[:minMatch] != sharedSuffixHeadPrefix || row[minMatch:2*minMatch] != sharedSuffixHead {
				t.Fatalf("row %d does not retain the shared equal-length suffix head", i)
			}
		}
		verifyAdversarialTrainFixture(t, rows, variants)
	}
}

type adversarialEncodeFixture struct {
	model         *Model
	rows          []string
	totalBytes    int64
	expectedCodes int
	slotProbes    int
}

func newAdversarialEncodeFixture(tb testing.TB, variants, slotCount int, colliding bool) adversarialEncodeFixture {
	tb.Helper()
	heads := adversarialHeads(slotCount, variants+1, colliding)
	verifyAdversarialHeadShape(tb, heads, slotCount, colliding)

	matcher := newMatcher(0)
	dictionary := make([]byte, 0, singleByteTokens+variants*(2*minMatch+len(adversarialLongBucketTail)))
	boundaries := make([]uint32, 0, singleByteTokens+variants+1)
	boundaries = append(boundaries, 0)
	for i := 0; i < singleByteTokens; i++ {
		entry := []byte{byte(i)}
		if !matcher.insert(entry, uint16(i)) {
			tb.Fatalf("insert byte token %d", i)
		}
		dictionary = append(dictionary, entry...)
		boundaries = append(boundaries, uint32(len(dictionary)))
	}

	rows := make([]string, 0, variants*2)
	for i, head := range heads[:variants] {
		entry := adversarialToken(head)
		if !matcher.insert(entry, uint16(singleByteTokens+i)) {
			tb.Fatalf("insert long token %d", i)
		}
		dictionary = append(dictionary, entry...)
		boundaries = append(boundaries, uint32(len(dictionary)))
		rows = append(rows, string(entry))
	}
	reject := string(adversarialToken(heads[variants]))
	for range variants {
		rows = append(rows, reject)
	}

	model := &Model{matcher: matcher, dictionary: dictionary, tokenBoundaries: boundaries}
	archive, err := model.Encode(rows)
	if err != nil {
		tb.Fatalf("Model.Encode setup: %v", err)
	}
	for _, index := range []int{0, variants, len(rows) - 1} {
		got, err := archive.AppendRow(nil, index)
		if err != nil {
			tb.Fatalf("AppendRow(%d): %v", index, err)
		}
		if string(got) != rows[index] {
			tb.Fatalf("round trip row %d", index)
		}
	}

	var totalBytes int64
	for _, row := range rows {
		totalBytes += int64(len(row))
	}
	return adversarialEncodeFixture{
		model:         model,
		rows:          rows,
		totalBytes:    totalBytes,
		expectedCodes: len(archive.CompressedData),
		slotProbes:    adversarialSlotProbes(heads, slotCount),
	}
}

func benchmarkModelEncodeLongBucketHeads(b *testing.B, variants, slotCount int, colliding bool) {
	fixture := newAdversarialEncodeFixture(b, variants, slotCount, colliding)
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
	b.ReportMetric(float64(fixture.slotProbes), "preimage-slot-probes/op")
}

func BenchmarkModelEncodeLongBucketHashFloodLarge(b *testing.B) {
	benchmarkModelEncodeLongBucketHeads(b, 273, 640, true)
}

func BenchmarkModelEncodeLongBucketHashFloodExpanded(b *testing.B) {
	benchmarkModelEncodeLongBucketHeads(b, 8192, 10000, true)
}

func BenchmarkModelEncodeLongBucketDistributedLarge(b *testing.B) {
	benchmarkModelEncodeLongBucketHeads(b, 273, 640, false)
}

func BenchmarkModelEncodeLongBucketDistributedExpanded(b *testing.B) {
	benchmarkModelEncodeLongBucketHeads(b, 8192, 10000, false)
}

func benchmarkModelTrainLongBucketHeads(b *testing.B, variants, slotCount int, colliding bool) {
	rows, heads := adversarialCollisionRows(variants, slotCount, colliding)
	verifyAdversarialHeadShape(b, heads, slotCount, colliding)
	verifyAdversarialTrainFixture(b, rows, variants)

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
			b.Fatal("Model.Train returned nil")
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*int(totalBytes)), "ns/byte")
	b.ReportMetric(float64(adversarialSlotProbes(heads, slotCount)), "preimage-slot-probes/op")
}

func BenchmarkModelTrainLongBucketHashFloodLarge(b *testing.B) {
	benchmarkModelTrainLongBucketHeads(b, 273, 640, true)
}

func BenchmarkModelTrainLongBucketHashFloodExpanded(b *testing.B) {
	benchmarkModelTrainLongBucketHeads(b, 8192, 10000, true)
}

func BenchmarkModelTrainLongBucketDistributedLarge(b *testing.B) {
	benchmarkModelTrainLongBucketHeads(b, 273, 640, false)
}

func BenchmarkModelTrainLongBucketDistributedExpanded(b *testing.B) {
	benchmarkModelTrainLongBucketHeads(b, 8192, 10000, false)
}

func BenchmarkModelTrainSharedSuffixHeadReverseExpandedVerified(b *testing.B) {
	rows := reverseTrainingDiscoveryOrder(sharedSuffixHeadRows(8192))
	verifyAdversarialTrainFixture(b, rows, 8192)
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
