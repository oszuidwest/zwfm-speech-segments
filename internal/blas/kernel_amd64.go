package blas

import "golang.org/x/sys/cpu"

// init selects the AVX2/FMA kernel when the CPU has both; otherwise the Go
// kernel stays.
func init() {
	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		best = kernel{6, 16, kernelAVX2}
	}
}

// kernel6x16 is in kernel_amd64.s: 12 YMM accumulators hold a 6×16 block.
//
//go:noescape
func kernel6x16(k int, a, b, c *float32, ldc int)

// kernelAVX2 checks the slice bounds the assembly relies on, then runs
// kernel6x16.
func kernelAVX2(pk int, ap, bp, c []float32, ldc int) {
	_, _, _ = ap[pk*6-1], bp[pk*16-1], c[5*ldc+15] // the assembly reads and writes this much
	kernel6x16(pk, &ap[0], &bp[0], &c[0], ldc)
}
