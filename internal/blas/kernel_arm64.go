package blas

// init selects the NEON kernel. ASIMD (NEON) is part of every arm64 CPU, so no
// feature check is needed.
func init() {
	best = kernel{8, 12, kernelNEON}
}

// kernel8x12 is in kernel_arm64.s: 24 NEON accumulators hold an 8×12 block.
// The Go assembler has no by-element FMLA, so those are encoded with WORD.
//
//go:noescape
func kernel8x12(k int, a, b, c *float32, ldc int)

// kernelNEON checks the slice bounds the assembly relies on, then runs
// kernel8x12.
func kernelNEON(pk int, ap, bp, c []float32, ldc int) {
	_, _, _ = ap[pk*8-1], bp[pk*12-1], c[7*ldc+11] // the assembly reads and writes this much
	kernel8x12(pk, &ap[0], &bp[0], &c[0], ldc)
}
