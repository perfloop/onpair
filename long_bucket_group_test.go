package onpair

import "testing"

const mixedSuffixHeadPrefix = "abcdefgh"

type mixedSuffixHeadFixture struct {
	model         *Model
	rows          []string
	totalBytes    int64
	expectedCodes int
}

// mixedSuffixHeadRows places entries from every suffix-head length class in one
// primary 8-byte bucket. The shorter probe shares a one-byte head with a
// longer candidate whose tail deliberately differs; the matcher must reject
// that tail and fall back to the shorter token.
func newMixedSuffixHeadFixture(tb testing.TB, bucketSize int) mixedSuffixHeadFixture {
	tb.Helper()
	if bucketSize < 22 {
		tb.Fatalf("bucket size %d is too small for mixed suffix-head classes", bucketSize)
	}

	matcher := newMatcher(0)
	dictionary := make([]byte, 0, singleByteTokens+bucketSize*(minMatch+8))
	boundaries := make([]uint32, 0, singleByteTokens+bucketSize+1)
	boundaries = append(boundaries, 0)
	for i := 0; i < singleByteTokens; i++ {
		entry := []byte{byte(i)}
		if !matcher.insert(entry, uint16(i)) {
			tb.Fatalf("insert byte token %d", i)
		}
		dictionary = append(dictionary, entry...)
		boundaries = append(boundaries, uint32(len(dictionary)))
	}

	shortSuffix := []byte{'@'}
	longSuffix := []byte("@ABCDEFGHtail")
	suffixes := make([][]byte, 0, bucketSize)
	suffixes = append(suffixes, shortSuffix)
	ordinal := 1
	for headLen := 2; headLen < minMatch; headLen++ {
		for range 3 {
			suffixes = append(suffixes, mixedSuffixHeadBytes(ordinal, headLen))
			ordinal++
		}
	}
	suffixes = append(suffixes, longSuffix)
	for len(suffixes) < bucketSize {
		suffixLen := minMatch + ordinal%5
		suffixes = append(suffixes, mixedSuffixHeadBytes(ordinal, suffixLen))
		ordinal++
	}

	rows := make([]string, 0, len(suffixes)+3)
	for i, suffix := range suffixes {
		entry := append([]byte(mixedSuffixHeadPrefix), suffix...)
		id := uint16(singleByteTokens + i)
		if !matcher.insert(entry, id) {
			tb.Fatalf("insert long token %d", i)
		}
		dictionary = append(dictionary, entry...)
		boundaries = append(boundaries, uint32(len(dictionary)))
		rows = append(rows, string(entry))
	}

	shorterProbeBytes := append([]byte(mixedSuffixHeadPrefix), longSuffix[:minMatch]...)
	shorterProbeBytes = append(shorterProbeBytes, []byte("MISS!")...)
	shorterProbe := string(shorterProbeBytes)
	rows = append(rows, shorterProbe)
	rejectProbe := append([]byte(mixedSuffixHeadPrefix), []byte{0xfe, 0xed, 0xfa, 0xce, 1, 2, 3, 4, 5}...)
	rows = append(rows, string(rejectProbe))
	rows = append(rows, string(append([]byte(mixedSuffixHeadPrefix), shortSuffix...)))

	model := &Model{
		matcher:         matcher,
		dictionary:      dictionary,
		tokenBoundaries: boundaries,
	}
	archive, err := model.Encode(rows)
	if err != nil {
		tb.Fatalf("Model.Encode setup: %v", err)
	}
	for i, want := range rows {
		got, err := archive.AppendRow(nil, i)
		if err != nil {
			tb.Fatalf("AppendRow(%d): %v", i, err)
		}
		if string(got) != want {
			tb.Fatalf("round trip row %d: got %q want %q", i, got, want)
		}
	}
	if len(archive.CompressedData) == 0 || archive.CompressedData[len(suffixes)] != singleByteTokens {
		tb.Fatalf("shorter probe did not select the one-byte suffix token")
	}

	var totalBytes int64
	for _, row := range rows {
		totalBytes += int64(len(row))
	}
	return mixedSuffixHeadFixture{
		model:         model,
		rows:          rows,
		totalBytes:    totalBytes,
		expectedCodes: len(archive.CompressedData),
	}
}

func mixedSuffixHeadBytes(ordinal, length int) []byte {
	suffix := make([]byte, length)
	value := uint64(ordinal)
	for i := 0; i < len(suffix) && i < minMatch; i++ {
		suffix[i] = byte(value)
		value >>= 8
	}
	for i := minMatch; i < len(suffix); i++ {
		suffix[i] = byte(ordinal*31 + i)
	}
	return suffix
}

func TestOnPair16SharedPrefixGroupsAndRoundTrip(t *testing.T) {
	matcher := newMatcher(16)
	tokens := []string{
		"abcdefghA",
		"abcdefghABC",
		"abcdefghABCD",
		"abcdefghABCE",
		"abcdefghABCDEFGH",
		"abcdefghZ",
	}
	for i, token := range tokens {
		if !matcher.insert([]byte(token), uint16(singleByteTokens+i)) {
			t.Fatalf("insert %q", token)
		}
	}

	cases := []struct {
		input string
		id    uint16
		len   int
	}{
		{"abcdefghABCDEFGH/next", singleByteTokens + 4, len(tokens[4])},
		{"abcdefghABCE/next", singleByteTokens + 3, len(tokens[3])},
		{"abcdefghABCD/next", singleByteTokens + 2, len(tokens[2])},
		{"abcdefghABC/next", singleByteTokens + 1, len(tokens[1])},
		{"abcdefghA/next", singleByteTokens, len(tokens[0])},
	}
	for _, tc := range cases {
		id, length, ok := matcher.find([]byte(tc.input))
		if !ok || id != tc.id || length != tc.len {
			t.Fatalf("find(%q): got id=%d length=%d ok=%v want id=%d length=%d", tc.input, id, length, ok, tc.id, tc.len)
		}
	}

	rows := sharedPrefixMatchRows(16)
	model, err := TrainModel(rows, WithThreshold(2), WithMaxTokenLength(16))
	if err != nil {
		t.Fatalf("TrainModel: %v", err)
	}
	if got := sharedPrefixMatchBucketLen(model); got < 2 {
		t.Fatalf("shared-prefix onPair16 bucket size: got %d want at least 2", got)
	}
	archive, err := model.Encode(rows)
	if err != nil {
		t.Fatalf("Model.Encode: %v", err)
	}
	for i, want := range rows {
		got, err := archive.AppendRow(nil, i)
		if err != nil {
			t.Fatalf("AppendRow(%d): %v", i, err)
		}
		if string(got) != want {
			t.Fatalf("round trip row %d: got %q want %q", i, got, want)
		}
	}
}

func BenchmarkModelEncodeMixedSuffixHeadSmallBucket(b *testing.B) {
	benchmarkModelEncodeMixedSuffixHeadBucket(b, 71)
}

func BenchmarkModelEncodeMixedSuffixHeadLargeBucket(b *testing.B) {
	benchmarkModelEncodeMixedSuffixHeadBucket(b, 273)
}

func benchmarkModelEncodeMixedSuffixHeadBucket(b *testing.B, bucketSize int) {
	fixture := newMixedSuffixHeadFixture(b, bucketSize)
	alternateRows := append([]string(nil), fixture.rows...)
	for i, j := 0, len(alternateRows)-1; i < j; i, j = i+1, j-1 {
		alternateRows[i], alternateRows[j] = alternateRows[j], alternateRows[i]
	}

	b.ReportAllocs()
	b.SetBytes(fixture.totalBytes)
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		rows := fixture.rows
		if i&1 != 0 {
			rows = alternateRows
		}
		archive, err := fixture.model.Encode(rows)
		if err != nil {
			b.Fatalf("Model.Encode: %v", err)
		}
		if got := len(archive.CompressedData); got != fixture.expectedCodes {
			b.Fatalf("compressed token count: got %d want %d", got, fixture.expectedCodes)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*int(fixture.totalBytes)), "ns/byte")
}

// sharedPrefixRowsInTrainingOrder reverses the actual order seen by
// Encoder.train after its deterministic shuffle. This makes the bucket's
// suffix heads arrive in reverse input order rather than merely reversing the
// caller slice before the shuffler rearranges it again.
func sharedPrefixRowsInTrainingOrder(rows []string) []string {
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

func BenchmarkModelTrainSharedPrefixReverseHeadsLargeBucket(b *testing.B) {
	benchmarkModelTrainSharedPrefixReverseHeadsBucket(b, 256, 287)
}

func BenchmarkModelTrainSharedPrefixReverseHeadsExpandedBucket(b *testing.B) {
	// 8,192 variants produce an 8,732-entry bucket while keeping the 720,896 B
	// corpus below the documented default 1 MiB training sample cap.
	benchmarkModelTrainSharedPrefixReverseHeadsBucket(b, 8192, 8732)
}

func benchmarkModelTrainSharedPrefixReverseHeadsBucket(b *testing.B, variants, wantBucketLen int) {
	rows := sharedPrefixRowsInTrainingOrder(sharedPrefixMatchRows(variants))
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
