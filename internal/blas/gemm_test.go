package blas

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestGemm checks the pure Go GEMM with every kernel this CPU can run
// against a float64 reference, on sizes that leave partial tiles and span
// several kc blocks, with beta 0 (C holds NaN) and beta 1.
func TestGemm(t *testing.T) {
	t.Logf("best kernel on this CPU: %dx%d", best.mr, best.nr)
	kernels := map[string]kernel{"4x4": {4, 4, kernel4x4}, "best": best}
	sizes := [][3]int{{1, 1, 1}, {7, 5, 3}, {13, 37, 300}, {6, 16, 256}, {61, 129, 529}}
	rng := rand.New(rand.NewPCG(1, 2)) //nolint:gosec // G404: fixed-seed test data, not security
	for name, kn := range kernels {
		for _, s := range sizes {
			m, n, k := s[0], s[1], s[2]
			a, b := make([]float32, m*k), make([]float32, k*n)
			for i := range a {
				a[i] = rng.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rng.Float32()*2 - 1
			}
			for _, beta := range []float32{0, 1} {
				c := make([]float32, m*n)
				for i := range c {
					c[i] = float32(math.NaN())
					if beta == 1 {
						c[i] = float32(i % 3)
					}
				}
				gemm(kn, m, n, k, a, b, beta, c)
				for i := range m {
					for j := range n {
						want := 0.0
						if beta == 1 {
							want = float64((i*n + j) % 3)
						}
						for p := range k {
							want += float64(a[i*k+p]) * float64(b[p*n+j])
						}
						if got := float64(c[i*n+j]); math.Abs(got-want) > 1e-4*math.Sqrt(float64(k)) {
							t.Fatalf("%s %dx%dx%d beta %v: C[%d,%d] = %v, want %v", name, m, n, k, beta, i, j, got, want)
						}
					}
				}
			}
		}
	}
}

// BenchmarkGemm runs the shape of the heaviest x-vector layer.
func BenchmarkGemm(b *testing.B) {
	m, n, k := 36000, 512, 1536
	x, y, c := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n)
	for b.Loop() {
		gemm(best, m, n, k, x, y, 0, c)
	}
	b.ReportMetric(2*float64(m*n*k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
}
