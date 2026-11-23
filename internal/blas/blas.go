// Package blas multiplies float32 matrices in pure Go: a packed,
// cache-blocked driver with an assembly kernel per CPU (AVX2/FMA on amd64,
// NEON on arm64) and a register-blocked Go kernel elsewhere. No cgo and no
// external library are needed.
package blas

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
