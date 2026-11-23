package blas

import "github.com/oszuidwest/zwfm-speech-segments/internal/par"

// kernel adds the product of two packed panels to C: ap holds pk×mr values
// of A as [p][mr], bp holds pk×nr values of B as [p][nr], and row r of the
// mr×nr block of C starts at c[r*ldc]. The driver only passes full blocks;
// edge blocks go through a zeroed mr×nr buffer.
type kernel struct {
	mr, nr int
	fn     func(pk int, ap, bp, c []float32, ldc int)
}

// best is the fastest kernel for this CPU; kernel_<arch>.go may replace it.
var best = kernel{4, 4, kernel4x4}

const (
	kc = 256 // depth of a packed block: an nr-wide panel of B stays in L1
	mc = 96  // rows per task: their packed A block stays in L2
)

// gemm computes C = A·B + beta·C with kernel kn, in the GotoBLAS order: per
// kc-deep block, B is packed once and shared; each goroutine packs mc rows
// of A and runs the kernel over every nr-wide panel of B (outer) and every
// mr-row strip of its A block (inner), so the B panel is reused from L1.
func gemm(kn kernel, m, n, k int, a, b []float32, beta float32, c []float32) {
	c = c[:m*n]
	switch beta {
	case 1:
	case 0:
		clear(c)
	default:
		for i := range c {
			c[i] *= beta
		}
	}
	mr, nr := kn.mr, kn.nr
	strips := (n + nr - 1) / nr
	rowStrips := (m + mr - 1) / mr
	group := max(1, mc/mr) // row strips per task
	bp := make([]float32, strips*kc*nr)
	for p0 := 0; p0 < k; p0 += kc {
		pk := min(kc, k-p0)
		par.For(strips, func(s int) {
			dst := bp[s*kc*nr : s*kc*nr+pk*nr]
			j0 := s * nr
			w := min(nr, n-j0)
			for p := range pk {
				row := b[(p0+p)*n+j0 : (p0+p)*n+j0+w]
				d := dst[p*nr : p*nr+nr]
				copy(d, row)
				clear(d[w:])
			}
		})
		par.For((rowStrips+group-1)/group, func(t int) {
			first, last := t*group, min((t+1)*group, rowStrips)
			ap := make([]float32, (last-first)*pk*mr)
			tile := make([]float32, mr*nr)
			for rs := first; rs < last; rs++ {
				dst := ap[(rs-first)*pk*mr : (rs-first+1)*pk*mr]
				i0 := rs * mr
				for r := range mr {
					if i0+r < m {
						for p, v := range a[(i0+r)*k+p0 : (i0+r)*k+p0+pk] {
							dst[p*mr+r] = v
						}
					} else {
						for p := range pk {
							dst[p*mr+r] = 0
						}
					}
				}
			}
			for s := range strips {
				bs := bp[s*kc*nr : s*kc*nr+pk*nr]
				j0 := s * nr
				w := min(nr, n-j0)
				for rs := first; rs < last; rs++ {
					as := ap[(rs-first)*pk*mr : (rs-first+1)*pk*mr]
					i0 := rs * mr
					h := min(mr, m-i0)
					if h == mr && w == nr {
						kn.fn(pk, as, bs, c[i0*n+j0:], n)
						continue
					}
					clear(tile)
					kn.fn(pk, as, bs, tile, nr)
					for r := range h {
						dst := c[(i0+r)*n+j0 : (i0+r)*n+j0+w]
						for j, v := range tile[r*nr : r*nr+w] {
							dst[j] += v
						}
					}
				}
			}
		})
	}
}

// kernel4x4 is the pure Go fallback for other architectures. Sixteen
// accumulators stay in registers; amd64 and arm64 use assembly kernels.
func kernel4x4(pk int, ap, bp, c []float32, ldc int) {
	var c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33 float32
	ap, bp = ap[:pk*4], bp[:pk*4]
	for p := 0; p+4 <= len(ap) && p+4 <= len(bp); p += 4 {
		a := ap[p : p+4 : p+4]
		b := bp[p : p+4 : p+4]
		a0, a1, a2, a3 := a[0], a[1], a[2], a[3]
		b0, b1, b2, b3 := b[0], b[1], b[2], b[3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3
	}
	r0 := c[0:4:4]
	r1 := c[ldc : ldc+4 : ldc+4]
	r2 := c[2*ldc : 2*ldc+4 : 2*ldc+4]
	r3 := c[3*ldc : 3*ldc+4 : 3*ldc+4]
	r0[0], r0[1], r0[2], r0[3] = r0[0]+c00, r0[1]+c01, r0[2]+c02, r0[3]+c03
	r1[0], r1[1], r1[2], r1[3] = r1[0]+c10, r1[1]+c11, r1[2]+c12, r1[3]+c13
	r2[0], r2[1], r2[2], r2[3] = r2[0]+c20, r2[1]+c21, r2[2]+c22, r2[3]+c23
	r3[0], r3[1], r3[2], r3[3] = r3[0]+c30, r3[1]+c31, r3[2]+c32, r3[3]+c33
}
