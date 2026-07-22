//go:build perfprobe

package onpair

const pairCounterProbeHistogramBuckets = 64

type pairCounterProbeStats struct {
	calls        uint64
	totalSteps   uint64
	histogram    [pairCounterProbeHistogramBuckets]uint64
	overflow     uint64
	maxSteps     int
	peakUsed     int
	peakCapacity int
}

var pairCounterProbe pairCounterProbeStats

func resetPairCounterProbe() {
	pairCounterProbe = pairCounterProbeStats{}
}

func recordPairCounterProbe(steps, used, capacity int) {
	pairCounterProbe.calls++
	pairCounterProbe.totalSteps += uint64(steps)
	if steps < len(pairCounterProbe.histogram) {
		pairCounterProbe.histogram[steps]++
	} else {
		pairCounterProbe.overflow++
	}
	if steps > pairCounterProbe.maxSteps {
		pairCounterProbe.maxSteps = steps
	}
	if capacity > 0 && (pairCounterProbe.peakCapacity == 0 ||
		used*pairCounterProbe.peakCapacity > pairCounterProbe.peakUsed*capacity) {
		pairCounterProbe.peakUsed = used
		pairCounterProbe.peakCapacity = capacity
	}
}

func snapshotPairCounterProbe() pairCounterProbeStats {
	return pairCounterProbe
}
