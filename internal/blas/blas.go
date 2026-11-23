// Package blas multiplies float32 matrices in pure Go: a packed,
// cache-blocked driver with an assembly kernel per CPU (AVX2/FMA on amd64,
// NEON on arm64) and a register-blocked Go kernel elsewhere. No cgo and no
// external library are needed.
package blas

// BlockDepth is the depth in which Sgemm adds up products. Splitting the
// depth of a product into consecutive parts that are multiples of
// BlockDepth, and accumulating the parts with beta 1, gives exactly the same
// float32 result as one Sgemm over the whole depth.
const BlockDepth = kc

// Sgemm computes C = A·B + beta·C for row-major A (m×k), B (k×n) and C (m×n).
// With beta 0, C's old contents are ignored.
func Sgemm(m, n, k int, a, b []float32, beta float32, c []float32) {
	if m == 0 || n == 0 {
		return
	}
	if len(a) < m*k || len(b) < k*n || len(c) < m*n {
		panic("blas: Sgemm slice too short")
	}
	gemm(best, m, n, k, a, b, beta, c)
}
