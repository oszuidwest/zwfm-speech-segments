// Adapted from gonum.org/v1/gonum/lapack/gonum (dsyev.go, dsteqr.go).
//
// Copyright ©2016 The Gonum Authors. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Gonum project nor the names of its authors and
//       contributors may be used to endorse or promote products derived from this
//       software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package diarize

import (
	"math"
	"slices"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/gonum"
	"gonum.org/v1/gonum/mat"
)

// eigDesc returns eigenvalues from greatest to smallest and the matching
// eigenvectors as rows (Kaldi Eig followed by SortSvd). It is gonum's Dsyev
// with the eigenvectors kept as rows, so that Dsteqr's rotations run along
// contiguous memory.
func eigDesc(s *mat.SymDense) ([]float64, *mat.Dense) {
	n := s.SymmetricDim()
	a := mat.DenseCopyOf(s)
	raw := a.RawMatrix()
	d, e, tau := make([]float64, n), make([]float64, max(n-1, 1)), make([]float64, max(n-1, 1))
	impl := gonum.Implementation{}
	// Dsyev scales a matrix with extreme entries into a safe range.
	const rmin, rmax = 0x1p-485, 0x1p485 // sqrt(dlamchS/dlamchP) and its inverse
	sigma := 1.0
	switch anrm := impl.Dlansy(lapack.MaxAbs, blas.Upper, n, raw.Data, raw.Stride, nil); {
	case anrm > 0 && anrm < rmin:
		sigma = rmin / anrm
	case anrm > rmax:
		sigma = rmax / anrm
	}
	if sigma != 1 {
		impl.Dlascl(lapack.UpperTri, 0, 0, 1, sigma, n, n, raw.Data, raw.Stride)
	}
	work := []float64{0}
	impl.Dsytrd(blas.Upper, n, raw.Data, raw.Stride, d, e, tau, work, -1)
	work = make([]float64, max(int(work[0]), 2*n))
	impl.Dsytrd(blas.Upper, n, raw.Data, raw.Stride, d, e, tau, work, len(work))
	impl.Dorgtr(blas.Upper, n, raw.Data, raw.Stride, tau, work, len(work))
	var zt mat.Dense
	zt.CloneFrom(a.T())
	if !dsteqrRows(n, d, e, zt.RawMatrix().Data, zt.RawMatrix().Stride, work) {
		panic("diarize: eigen decomposition failed")
	}
	for i := range n { // a is free again; reuse it for the descending rows
		a.SetRow(i, zt.RawRowView(n-1-i))
	}
	floats.Scale(1/sigma, d)
	slices.Reverse(d)
	return d, a
}

// dsteqrRows is gonum's Dsteqr with compz == lapack.EVOrig, except that zt
// holds the transpose of z: on exit its rows are the eigenvectors. Each
// plane rotation then updates two rows instead of two strided columns.
// The rotations and their arithmetic are those of Dsteqr.
//
//nolint:gocyclo // kept line by line with gonum's Dsteqr for review
func dsteqrRows(n int, d, e, zt []float64, ldz int, work []float64) (ok bool) {
	if n <= 1 {
		return true
	}
	impl := gonum.Implementation{}

	const (
		eps    = 0x1p-53   // dlamchE
		safmin = 0x1p-1022 // dlamchS
	)
	eps2 := eps * eps
	safmax := 1 / safmin
	ssfmax := math.Sqrt(safmax) / 3
	ssfmin := math.Sqrt(safmin) / eps2

	const maxit = 30
	nmaxit := n * maxit

	jtot := 0

	// Determine where the matrix splits and choose QL or QR iteration for each
	// block, according to whether top or bottom diagonal element is smaller.
	l1 := 0
	nm1 := n - 1

	type scaletype int
	const (
		down scaletype = iota + 1
		up
	)
	for {
		if l1 > n-1 {
			// Order eigenvalues and eigenvectors.
			for ii := 1; ii < n; ii++ {
				i := ii - 1
				k := i
				p := d[i]
				for j := ii; j < n; j++ {
					if d[j] < p {
						k = j
						p = d[j]
					}
				}
				if k != i {
					d[k] = d[i]
					d[i] = p
					zi, zk := zt[i*ldz:i*ldz+n], zt[k*ldz:k*ldz+n]
					for j := range zi {
						zi[j], zk[j] = zk[j], zi[j]
					}
				}
			}
			return true
		}
		if l1 > 0 {
			e[l1-1] = 0 //nolint:gosec // G602: 0 < l1 <= n-1 here and len(e) >= n-1
		}
		var m int
		if l1 <= nm1 {
			for m = l1; m < nm1; m++ {
				test := math.Abs(e[m])
				if test == 0 {
					break
				}
				if test <= (math.Sqrt(math.Abs(d[m]))*math.Sqrt(math.Abs(d[m+1])))*eps {
					e[m] = 0
					break
				}
			}
		}
		l := l1
		lsv := l
		lend := m
		lendsv := lend
		l1 = m + 1
		if lend == l {
			continue
		}

		// Scale submatrix in rows and columns L to Lend. Like LAPACK
		// (ISCALE = 0), every block starts unscaled; gonum v0.17.0 keeps the
		// flag of an earlier block and would wrongly unscale this one.
		var iscale scaletype
		anorm := impl.Dlanst(lapack.MaxAbs, lend-l+1, d[l:], e[l:])
		switch {
		case anorm == 0:
			continue
		case anorm > ssfmax:
			iscale = down
			// Pretend that d and e are matrices with 1 column.
			impl.Dlascl(lapack.General, 0, 0, anorm, ssfmax, lend-l+1, 1, d[l:], 1)
			impl.Dlascl(lapack.General, 0, 0, anorm, ssfmax, lend-l, 1, e[l:], 1)
		case anorm < ssfmin:
			iscale = up
			impl.Dlascl(lapack.General, 0, 0, anorm, ssfmin, lend-l+1, 1, d[l:], 1)
			impl.Dlascl(lapack.General, 0, 0, anorm, ssfmin, lend-l, 1, e[l:], 1)
		}

		// Choose between QL and QR.
		if math.Abs(d[lend]) < math.Abs(d[l]) {
			lend = lsv
			l = lendsv
		}
		if lend > l {
			// QL Iteration. Look for small subdiagonal element.
			for {
				if l != lend {
					for m = l; m < lend; m++ {
						v := math.Abs(e[m])
						if v*v <= (eps2*math.Abs(d[m]))*math.Abs(d[m+1])+safmin {
							break
						}
					}
				} else {
					m = lend
				}
				if m < lend {
					e[m] = 0
				}
				p := d[l]
				if m == l {
					// Eigenvalue found.
					l++
					if l > lend {
						break
					}
					continue
				}

				// If remaining matrix is 2×2, use Dlae2 to compute its eigensystem.
				if m == l+1 {
					d[l], d[l+1], work[l], work[n-1+l] = impl.Dlaev2(d[l], e[l], d[l+1])
					impl.Dlasr(blas.Left, lapack.Variable, lapack.Backward,
						2, n, work[l:], work[n-1+l:], zt[l*ldz:], ldz)
					e[l] = 0
					l += 2
					if l > lend {
						break
					}
					continue
				}

				if jtot == nmaxit {
					break
				}
				jtot++

				// Form shift
				g := (d[l+1] - p) / (2 * e[l])
				r := impl.Dlapy2(g, 1)
				g = d[m] - p + e[l]/(g+math.Copysign(r, g))
				s := 1.0
				c := 1.0
				p = 0.0

				// Inner loop
				for i := m - 1; i >= l; i-- {
					f := s * e[i]
					b := c * e[i]
					c, s, r = impl.Dlartg(g, f)
					if i != m-1 {
						e[i+1] = r
					}
					g = d[i+1] - p
					r = (d[i]-g)*s + 2*c*b
					p = s * r
					d[i+1] = g + p
					g = c*r - b

					// Save rotations.
					work[i] = c
					work[n-1+i] = -s
				}
				// Apply saved rotations.
				mm := m - l + 1
				impl.Dlasr(blas.Left, lapack.Variable, lapack.Backward,
					mm, n, work[l:], work[n-1+l:], zt[l*ldz:], ldz)
				d[l] -= p
				e[l] = g
			}
		} else {
			// QR Iteration.
			// Look for small superdiagonal element.
			for {
				if l != lend {
					for m = l; m > lend; m-- {
						v := math.Abs(e[m-1])
						if v*v <= (eps2*math.Abs(d[m])*math.Abs(d[m-1]) + safmin) {
							break
						}
					}
				} else {
					m = lend
				}
				if m > lend {
					e[m-1] = 0
				}
				p := d[l]
				if m == l {
					// Eigenvalue found
					l--
					if l < lend {
						break
					}
					continue
				}

				// If remaining matrix is 2×2, use Dlae2 to compute its eigenvalues.
				if m == l-1 {
					d[l-1], d[l], work[m], work[n-1+m] = impl.Dlaev2(d[l-1], e[l-1], d[l])
					impl.Dlasr(blas.Left, lapack.Variable, lapack.Forward,
						2, n, work[m:], work[n-1+m:], zt[(l-1)*ldz:], ldz)
					e[l-1] = 0
					l -= 2
					if l < lend {
						break
					}
					continue
				}
				if jtot == nmaxit {
					break
				}
				jtot++

				// Form shift.
				g := (d[l-1] - p) / (2 * e[l-1])
				r := impl.Dlapy2(g, 1)
				g = d[m] - p + (e[l-1])/(g+math.Copysign(r, g))
				s := 1.0
				c := 1.0
				p = 0.0

				// Inner loop.
				for i := m; i < l; i++ {
					f := s * e[i]
					b := c * e[i]
					c, s, r = impl.Dlartg(g, f)
					if i != m {
						e[i-1] = r
					}
					g = d[i] - p
					r = (d[i+1]-g)*s + 2*c*b
					p = s * r
					d[i] = g + p
					g = c*r - b

					// Save rotations.
					work[i] = c
					work[n-1+i] = s
				}

				// Apply saved rotations.
				mm := l - m + 1
				impl.Dlasr(blas.Left, lapack.Variable, lapack.Forward,
					mm, n, work[m:], work[n-1+m:], zt[m*ldz:], ldz)
				d[l] -= p
				e[l-1] = g
			}
		}

		// Undo scaling if necessary.
		switch iscale {
		case down:
			// Pretend that d and e are matrices with 1 column.
			impl.Dlascl(lapack.General, 0, 0, ssfmax, anorm, lendsv-lsv+1, 1, d[lsv:], 1)
			impl.Dlascl(lapack.General, 0, 0, ssfmax, anorm, lendsv-lsv, 1, e[lsv:], 1)
		case up:
			impl.Dlascl(lapack.General, 0, 0, ssfmin, anorm, lendsv-lsv+1, 1, d[lsv:], 1)
			impl.Dlascl(lapack.General, 0, 0, ssfmin, anorm, lendsv-lsv, 1, e[lsv:], 1)
		}

		// Check for no convergence to an eigenvalue after a total of n*maxit iterations.
		if jtot >= nmaxit {
			break
		}
	}
	for i := 0; i < n-1; i++ {
		if e[i] != 0 {
			return false
		}
	}
	return true
}
