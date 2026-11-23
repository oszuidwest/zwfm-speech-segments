package diarize

import (
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// TestEigDesc pins eigDesc against gonum's EigenSym on random symmetric
// matrices: the same eigenvalues in descending order, and unit rows v with
// A*v = lambda*v. The falling diagonal makes Dsteqr take its QR branch
// instead of QL; the huge scale needs Dsyev's rescaling.
func TestEigDesc(t *testing.T) {
	const n = 60
	r := rand.New(rand.NewPCG(1, 2)) //nolint:gosec // G404: seeded test data
	for _, tc := range []struct{ ramp, scale float64 }{{0, 1}, {1, 1}, {0, 1e200}} {
		a := mat.NewSymDense(n, nil)
		for i := range n {
			for j := i; j < n; j++ {
				a.SetSym(i, j, r.NormFloat64()*tc.scale)
			}
			a.SetSym(i, i, a.At(i, i)+tc.ramp*float64(n-i))
		}
		var es mat.EigenSym
		if !es.Factorize(a, false) {
			t.Fatal("EigenSym failed")
		}
		want := es.Values(nil)
		tol := 1e-10 * tc.scale
		vals, vecs := eigDesc(a)
		// The comparisons are written as !(x <= tol) so that NaN fails.
		for k, v := range vals {
			if w := want[n-1-k]; !(math.Abs(v-w) <= tol) {
				t.Errorf("%+v: eigenvalue %d = %g, want %g", tc, k, v, w)
			}
			row := vecs.RawRowView(k)
			if norm := floats.Norm(row, 2); !(math.Abs(norm-1) <= 1e-10) {
				t.Errorf("%+v: eigenvector %d has norm %g", tc, k, norm)
			}
			var av mat.VecDense
			av.MulVec(a, mat.NewVecDense(n, row))
			floats.AddScaled(av.RawVector().Data, -v, row)
			if res := floats.Norm(av.RawVector().Data, 2); !(res <= tol) {
				t.Errorf("%+v: eigenvector %d: |A*v - lambda*v| = %g", tc, k, res)
			}
		}
	}
}
