package onpair

// The equal-suffix fixture must keep the common-prefix token from producing
// short-prefix intermediates while it trains its long entries. A zero prefix
// gives the public TrainModel fixture exactly the documented shape: 2,040
// common-bucket entries, each with a two-byte suffix.
func init() {
	longBucketEqualSuffixPrefix = [minMatch]byte{}
}
