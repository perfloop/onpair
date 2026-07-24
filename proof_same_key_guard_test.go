package onpair

import (
	"fmt"
	"testing"
)

func proofSameKeyRows() []string {
	r := make([]string, 8192*4)
	for i := range r {
		r[i] = fmt.Sprintf("abcdefghABCDEFGHIJKLMNOP%04xxy", i/4)
	}
	return r
}

func BenchmarkProofSharedSuffixKeyRejectExpanded(b *testing.B) {
	r := proofSameKeyRows()
	m, err := TrainModel(r, WithThreshold(2))
	if err != nil {
		b.Fatal(err)
	}
	for i := range r {
		r[i] = "abcdefghABCDEFGHIJKLMNOPffffxy"
	}
	a, err := m.Encode(r)
	if err != nil {
		b.Fatal(err)
	}
	want := len(a.CompressedData)
	b.ReportAllocs()
	b.SetBytes(int64(len(r) * len(r[0])))
	b.ResetTimer()
	for b.Loop() {
		a, err := m.Encode(r)
		if err != nil || len(a.CompressedData) != want {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*len(r)*len(r[0])), "ns/byte")
}
