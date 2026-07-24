package onpair

import "testing"

const longTailRejectPrefix = "abcdefgh"

type longTailRejectFixture struct {
	model           *Model
	rows            []string
	totalBytes      int64
	expectedCodes   int
	lengthGroups    int
	groupSlotProbes int
	chainVisits     int
}

func longTailGroupSlotProbes(bucket *longBucket, key uint64, keyLen int) (int, int) {
	slot := longBucketGroupStartSlot(key, keyLen, len(bucket.groupSlots))
	probes := 0
	for {
		probes++
		entry := bucket.groupSlots[slot]
		if entry == 0 {
			return probes, 0
		}
		candidate := &bucket.candidates[entry-1]
		if int(candidate.suffixLen) == keyLen && candidate.key == key {
			visits := 0
			for i := int(entry - 1); i >= 0; {
				visits++
				if bucket.previous == nil {
					break
				}
				i = int(bucket.previous[i])
			}
			return probes, visits
		}
		slot++
		if slot == len(bucket.groupSlots) {
			slot = 0
		}
	}
}

func newLongTailRejectFixture(tb testing.TB, variants, tailLen int) longTailRejectFixture {
	tb.Helper()
	matcher := newMatcher(0)
	dictionary := make([]byte, 0, singleByteTokens+variants*96)
	boundaries := make([]uint32, 0, singleByteTokens+variants+1)
	boundaries = append(boundaries, 0)
	for i := 0; i < singleByteTokens; i++ {
		if !matcher.insert([]byte{byte(i)}, uint16(i)) {
			tb.Fatalf("insert byte token %d", i)
		}
		dictionary = append(dictionary, byte(i))
		boundaries = append(boundaries, uint32(len(dictionary)))
	}
	for i := 0; i < variants; i++ {
		suffixLen := 17 + i%64
		entry := make([]byte, minMatch+suffixLen)
		copy(entry, longTailRejectPrefix)
		entry[minMatch] = 'a'
		for j := minMatch + 1; j < len(entry); j++ {
			entry[j] = byte(1 + (i*31+j*17)%251)
		}
		if !matcher.insert(entry, uint16(singleByteTokens+i)) {
			tb.Fatalf("insert long token %d", i)
		}
		dictionary = append(dictionary, entry...)
		boundaries = append(boundaries, uint32(len(dictionary)))
	}
	bucket := matcher.longMatchBuckets.get(bytesToU64LE([]byte(longTailRejectPrefix), minMatch))
	if bucket == nil || bucket.groupLenBits == nil {
		tb.Fatal("64 suffix lengths did not promote the bucket length bitmap")
	}
	reject := make([]byte, minMatch+tailLen)
	copy(reject, longTailRejectPrefix)
	for i := minMatch; i < len(reject); i++ {
		reject[i] = byte(33 + i%31)
	}
	if reject[minMatch] == 'a' {
		tb.Fatal("rejecting suffix head matches a trained suffix")
	}
	rows := make([]string, variants)
	for i := range rows {
		rows[i] = string(reject)
	}
	model := &Model{matcher: matcher, dictionary: dictionary, tokenBoundaries: boundaries}
	archive, err := model.Encode(rows)
	if err != nil {
		tb.Fatalf("Model.Encode setup: %v", err)
	}
	for _, index := range []int{0, len(rows) / 2, len(rows) - 1} {
		got, err := archive.AppendRow(nil, index)
		if err != nil || string(got) != rows[index] {
			tb.Fatalf("round trip row %d: %v", index, err)
		}
	}
	maxSuffixLen := tailLen
	if maxSuffixLen > 80 {
		maxSuffixLen = 80
	}
	probes, visits := 0, 0
	for suffixLen := maxSuffixLen; suffixLen >= 17; suffixLen-- {
		key := longBucketGroupKey(reject[minMatch:minMatch+suffixLen], matcher.groupSeed)
		p, v := longTailGroupSlotProbes(bucket, key, suffixLen)
		probes += p
		visits += v
	}
	return longTailRejectFixture{
		model:           model,
		rows:            rows,
		totalBytes:      int64(len(rows) * len(reject)),
		expectedCodes:   len(archive.CompressedData),
		lengthGroups:    maxSuffixLen - 16,
		groupSlotProbes: probes,
		chainVisits:     visits,
	}
}

func TestLongTailRejectFixture(t *testing.T) {
	for _, tailLen := range []int{64, 1024, 4096} {
		fixture := newLongTailRejectFixture(t, 273, tailLen)
		if fixture.chainVisits != 0 || fixture.groupSlotProbes < fixture.lengthGroups {
			t.Fatalf("tail %d: chain visits %d, group-slot probes %d", tailLen, fixture.chainVisits, fixture.groupSlotProbes)
		}
	}
}

func benchmarkModelEncodeLongTailReject(b *testing.B, tailLen int) {
	fixture := newLongTailRejectFixture(b, 273, tailLen)
	b.ReportAllocs()
	b.SetBytes(fixture.totalBytes)
	b.ResetTimer()
	for b.Loop() {
		archive, err := fixture.model.Encode(fixture.rows)
		if err != nil || len(archive.CompressedData) != fixture.expectedCodes {
			b.Fatalf("Model.Encode: %v", err)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*int(fixture.totalBytes)), "ns/byte")
	b.ReportMetric(float64(fixture.groupSlotProbes), "group-slot-probes/row")
	b.ReportMetric(float64(fixture.chainVisits), "previous-chain-visits/row")
}

func BenchmarkModelEncodeLongTailReject64(b *testing.B) {
	benchmarkModelEncodeLongTailReject(b, 64)
}

func BenchmarkModelEncodeLongTailReject1024(b *testing.B) {
	benchmarkModelEncodeLongTailReject(b, 1024)
}

func BenchmarkModelEncodeLongTailReject4096(b *testing.B) {
	benchmarkModelEncodeLongTailReject(b, 4096)
}
