package diarize

import (
	"fmt"
	"math"
	"slices"

	"gonum.org/v1/gonum/mat"

	"github.com/oszuidwest/zwfm-speech-segments/internal/kaldi"
	"github.com/oszuidwest/zwfm-speech-segments/internal/par"
)

// plda is Kaldi's Plda (ivector/plda.cc) plus the within- and between-class
// covariances before diagonalisation, which Plda::ApplyTransform rebuilds
// for every recording; they do not depend on the recording, so Load
// computes them once.
type plda struct {
	mean            []float64
	within, between *mat.SymDense
}

// dim returns the PLDA dimension.
func (p plda) dim() int { return len(p.mean) }

// readPlda reads a binary Kaldi Plda object and derives the covariances.
func readPlda(path string) (plda, error) {
	r, f, err := kaldi.Open(path)
	if err != nil {
		return plda{}, err
	}
	defer func() { _ = f.Close() }() // read-only; a close error changes nothing
	var p plda
	if err := r.Expect("<Plda>"); err != nil {
		return p, err
	}
	if p.mean, err = r.Vector(); err != nil {
		return p, err
	}
	t, err := r.Matrix()
	if err != nil {
		return p, err
	}
	psi, err := r.Vector()
	if err != nil {
		return p, err
	}
	if err := r.Expect("</Plda>"); err != nil {
		return p, err
	}
	d := len(p.mean)
	if t.Rows != d || t.Cols != d || len(psi) != d {
		return p, fmt.Errorf("%s: inconsistent dims", path)
	}
	var inv mat.Dense
	if err := inv.Inverse(mat.NewDense(d, d, t.Data)); err != nil {
		return p, fmt.Errorf("%s: %w", path, err)
	}
	var scaled mat.Dense
	scaled.Apply(func(_, j int, v float64) float64 { return v * psi[j] }, &inv)
	p.within, p.between = symProduct(&inv, &inv), symProduct(&scaled, &inv)
	return p, nil
}

// symProduct returns a*b^T as a symmetric matrix (upper triangle).
func symProduct(a, b mat.Matrix) *mat.SymDense {
	var m mat.Dense
	m.Mul(a, b.T())
	r, _ := m.Dims()
	return mat.NewSymDense(r, m.RawMatrix().Data)
}

// preprocess is ivector-subtract-global-mean, transform-vec and
// ivector-normalize-length: one row per x-vector.
func (m *Model) preprocess(xvectors [][]float32) *mat.Dense {
	d := len(m.mean)
	x := mat.NewDense(len(xvectors), d, nil)
	for i, v := range xvectors {
		row := x.RawRowView(i)
		for k := range row {
			row[k] = float64(v[k]) - m.mean[k]
		}
	}
	var y mat.Dense
	y.Mul(x, mat.NewDense(m.lda.Rows, m.lda.Cols, m.lda.Data).T())
	for i := range len(xvectors) {
		row := y.RawRowView(i)
		if norm := mat.Norm(mat.NewVecDense(len(row), row), 2); norm != 0 {
			s := math.Sqrt(float64(len(row))) / norm
			for k := range row {
				row[k] *= s
			}
		}
	}
	return &y
}

// estPCA is EstPca from ivector-plda-scoring-dense.cc: the leading
// eigenvectors (as rows) of the recording's covariance. Like Kaldi it keeps
// one component more than needed to pass the energy target.
func estPCA(y *mat.Dense, target float64) *mat.Dense {
	n, d := y.Dims()
	mu := make([]float64, d)
	for i := range n {
		for k, v := range y.RawRowView(i) {
			mu[k] += v
		}
	}
	for k := range mu {
		mu[k] /= float64(n)
	}
	var yy mat.Dense
	yy.Mul(y.T(), y)
	cov := mat.NewSymDense(d, nil)
	for i := range d {
		for j := i; j < d; j++ {
			cov.SetSym(i, j, yy.At(i, j)/float64(n)-mu[i]*mu[j])
		}
	}
	vals, vecs := eigDesc(cov)
	var total, energy float64
	for _, v := range vals {
		total += v
	}
	dim := 1
	for energy/total <= target && dim <= d {
		energy += vals[dim-1]
		dim++
	}
	dim = min(dim, d)
	return mat.DenseCopyOf(vecs.Slice(0, d, 0, dim).T())
}

// eigDesc returns eigenvalues from greatest to smallest and the matching
// eigenvectors as columns (Kaldi Eig followed by SortSvd).
func eigDesc(s *mat.SymDense) ([]float64, *mat.Dense) {
	var es mat.EigenSym
	if !es.Factorize(s, true) {
		panic("diarize: eigen decomposition failed")
	}
	vals := es.Values(nil) // ascending
	var vecs mat.Dense
	es.VectorsTo(&vecs)
	d := len(vals)
	out := mat.NewDense(d, d, nil)
	for c := range d {
		out.SetCol(c, mat.Col(nil, d-1-c, &vecs))
	}
	slices.Reverse(vals)
	return vals, out
}

// scores is ivector-plda-scoring-dense --target-energy 0.9 for one
// recording: an n x n row-major matrix of PLDA log-likelihood ratios,
// rounded to float32 as Kaldi stores them.
func (m *Model) scores(xvectors [][]float32) []float32 {
	y := m.preprocess(xvectors)
	pca := estPCA(y, targetEnergy)
	t, psi := m.plda.transform(y, pca)
	return llrMatrix(t, psi)
}

// transform applies Plda::ApplyTransform(pca) and then TransformIvector
// (with its length normalisation) to each row of y, after the PCA.
func (p plda) transform(y, pca *mat.Dense) (transformed *mat.Dense, psi []float64) {
	dim, _ := pca.Dims()
	within, between := sandwich(pca, p.within), sandwich(pca, p.between)
	var ch mat.Cholesky
	if !ch.Factorize(within) {
		panic("diarize: within-class covariance not positive definite")
	}
	var l, t1 mat.TriDense
	ch.LTo(&l)
	if err := t1.InverseTri(&l); err != nil {
		panic(err)
	}
	psi, u := eigDesc(sandwich(&t1, between))
	for k := range psi {
		psi[k] = max(psi[k], 0)
	}
	// Map from the pre-PCA space: U^T C^{-1} P.
	var tr, full mat.Dense
	tr.Mul(u.T(), &t1)
	full.Mul(&tr, pca)
	var offset mat.VecDense
	offset.MulVec(&full, mat.NewVecDense(len(p.mean), p.mean))
	var out mat.Dense
	out.Mul(y, full.T())
	n, _ := out.Dims()
	for i := range n {
		row := out.RawRowView(i)
		var dot float64
		for k := range row {
			row[k] -= offset.AtVec(k)
			dot += row[k] * row[k] / (psi[k] + 1)
		}
		s := math.Sqrt(float64(dim) / dot)
		for k := range row {
			row[k] *= s
		}
	}
	return &out, psi
}

// sandwich returns a*s*a^T.
func sandwich(a mat.Matrix, s *mat.SymDense) *mat.SymDense {
	var as mat.Dense
	as.Mul(a, s)
	return symProduct(&as, a)
}

// llrMatrix is Plda::LogLikelihoodRatio with n=1 for all pairs of rows,
// expanded into a constant, a per-row term, a per-column term and a
// weighted dot product. Like Kaldi it computes in double precision, so
// scores near the clustering threshold fall on the same side.
func llrMatrix(t *mat.Dense, psi []float64) []float32 {
	n, dim := t.Dims()
	w := make([]float64, dim)
	sq := make([]float64, dim)
	var c float64
	for k, p := range psi {
		a := p / (p + 1)
		v := 1 + p/(p+1)
		w[k] = a / v
		sq[k] = 0.5/(p+1) - 0.5/v // weight of t_jk^2 on the test side
		c += 0.5*math.Log(p+1) - 0.5*math.Log(v)
	}
	tw := make([]float64, n*dim) // rows scaled by sqrt(w)
	rowTerm := make([]float64, n)
	testTerm := make([]float64, n)
	for i := range n {
		for k, v := range t.RawRowView(i) {
			tw[i*dim+k] = v * math.Sqrt(w[k])
			a := psi[k] / (psi[k] + 1)
			rowTerm[i] -= 0.5 * a * a * v * v / (1 + a)
			testTerm[i] += sq[k] * v * v
		}
	}
	out := make([]float32, n*n)
	par.For(n, func(i int) {
		ri := tw[i*dim : (i+1)*dim]
		for j := i; j < n; j++ {
			rj := tw[j*dim : (j+1)*dim]
			var d0, d1, d2, d3 float64
			k := 0
			for ; k+4 <= dim; k += 4 {
				d0 += ri[k] * rj[k]
				d1 += ri[k+1] * rj[k+1]
				d2 += ri[k+2] * rj[k+2]
				d3 += ri[k+3] * rj[k+3]
			}
			for ; k < dim; k++ {
				d0 += ri[k] * rj[k]
			}
			out[i*n+j] = float32(c + rowTerm[i] + testTerm[j] + (d0 + d1) + (d2 + d3))
			out[j*n+i] = float32(c + rowTerm[j] + testTerm[i] + (d0 + d1) + (d2 + d3))
		}
	})
	return out
}
