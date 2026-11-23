package xvector

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"strconv"

	"github.com/oszuidwest/zwfm-speech-segments/internal/blas"
	"github.com/oszuidwest/zwfm-speech-segments/internal/par"
)

// Window is one subsegment in seconds, as in the Kaldi subsegments file.
type Window struct{ Start, End float64 }

// Windows returns the recipe's subsegments for a recording of the given
// number of 10 ms frames: utils/data/get_uniform_subsegments.py with --max-segment-duration
// 1.5, --overlap-duration 0.75, --max-remaining-duration 0.5 and
// --constant-duration True on the single segment 0 to frames*0.01 seconds.
// Times are rounded to 3 decimals as the script prints them.
func Windows(frames int) []Window {
	const maxDur, overlap, maxRemaining = 1.5, 0.75, 0.5
	startTime, endTime := 0.0, float64(frames)/100
	var wins []Window
	add := func(s, e float64) {
		wins = append(wins, Window{round3(s), round3(e)})
	}
	dur := endTime - startTime
	start := startTime
	for dur > maxDur { // constant duration: the threshold is maxDur alone
		add(start-startTime, start-startTime+maxDur)
		start += maxDur - overlap
		dur -= maxDur - overlap
	}
	if dur < 0 {
		return wins
	}
	if dur < maxRemaining {
		start = max(endTime-maxDur, startTime)
	}
	end := min(start+maxDur, endTime)
	add(start-startTime, end-startTime)
	return wins
}

// round3 rounds x to 3 decimals through text, the way the script prints it.
func round3(x float64) float64 {
	v, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 3, 64), 64)
	return v
}

// frameRange is the [first, last] parent frame of a window as
// utils/data/subsegment_data_dir.sh and fix_subsegment_feats.pl cut it.
func frameRange(w Window, frames int) (first, last int) {
	first, last = int(w.Start/0.01+0.5), int(w.End/0.01-0.5)
	return first, min(last, frames-1)
}

// job is one window, which nnet3-xvector-compute runs as a single chunk. Its
// pooled tdnn5 frames are [p0, p1] in parent frames.
type job struct {
	n          int // frames in the window
	p0, p1     int
	sum, sumsq []float32
}

// Extract returns one x-vector (tdnn6.affine output) per window, in window
// order. feats is row-major frames x 30 (CMN features). Each window must hold
// min-chunk-size to max-chunk-size frames, so that Kaldi runs it as one chunk
// without padding; Extract panics otherwise.
//
// Kaldi runs the network on each window's frames alone and pools the tdnn5
// frames whose whole input context lies inside the window. Those frames do
// not depend on anything outside the window, so tdnn1..tdnn5 run once over
// the recording and each window sums its own range.
func (m *Model) Extract(frames int, feats []float32, wins []Window) [][]float32 {
	if len(feats) < frames*featDim {
		panic("xvector: feats shorter than frames x 30")
	}
	dim := m.layers[len(m.layers)-1].out
	jobs := make([]*job, len(wins))
	for wi, w := range wins {
		first, last := frameRange(w, frames)
		n := last - first + 1
		if n < m.minChunk || n > m.maxChunk {
			panic(fmt.Sprintf("xvector: window of %d frames outside the model's chunk sizes %d..%d",
				n, m.minChunk, m.maxChunk))
		}
		// Output t=0 pools tdnn5 frames t in [-poolL, poolR] that are
		// computable, i.e. whose input context lies in [0, n-1].
		jobs[wi] = &job{n: n, p0: first + max(m.ctxL, -m.poolL), p1: first + min(n-1-m.ctxR, m.poolR),
			sum: make([]float32, dim), sumsq: make([]float32, dim)}
	}
	m.accumulate(feats, dim, jobs)
	return m.finish(dim, jobs)
}

// blockFrames is how many tdnn5 frames one pass computes (about 110 MB of
// buffers).
const blockFrames = 4096

// accumulate sums x and x² of each job's pooled tdnn5 frames, adding frames in
// time order in float32 like Kaldi's AddRowRanges. Jobs are sorted by first
// pooled frame so each block only visits the windows that overlap it.
func (m *Model) accumulate(feats []float32, dim int, jobs []*job) {
	if len(jobs) == 0 {
		return
	}
	sorted := slices.SortedStableFunc(slices.Values(jobs), func(a, b *job) int { return a.p0 - b.p0 })
	lo, hi := sorted[0].p0, 0
	for _, j := range sorted {
		hi = max(hi, j.p1)
	}
	var s scratch
	next := 0 // sorted[:next] start before the current block
	var active []*job
	for b0 := lo; b0 <= hi; b0 += blockFrames {
		b1 := min(b0+blockFrames, hi+1)
		in := feats[(b0-m.ctxL)*featDim : (b1+m.ctxR)*featDim]
		out := m.frameLevel(in, b1-b0+m.ctxL+m.ctxR, &s)
		for next < len(sorted) && sorted[next].p0 < b1 {
			active = append(active, sorted[next])
			next++
		}
		par.For(len(active), func(i int) { addRows(active[i], out, dim, b0, b0, b1-1) })
		active = slices.DeleteFunc(active, func(j *job) bool { return j.p1 < b1 })
	}
}

// addRows adds rows [max(p0,lo), min(p1,hi)] of out, whose first row is
// frame base, to the job's sums.
func addRows(j *job, out []float32, dim, base, lo, hi int) {
	for t := max(j.p0, lo); t <= min(j.p1, hi); t++ {
		row := out[(t-base)*dim : (t-base+1)*dim]
		for i, x := range row {
			j.sum[i] += x
			j.sumsq[i] += float32(x * x)
		}
	}
}

// finish turns the sums into mean and stddev (StatisticsPoolingComponent) and
// applies tdnn6.affine.
func (m *Model) finish(dim int, jobs []*job) [][]float32 {
	in := m.tdnn6.in
	x := make([]float32, len(jobs)*in)
	for k, j := range jobs {
		row := x[k*in : (k+1)*in]
		count := float32(j.p1 - j.p0 + 1)
		for i := range m.logCounts {
			row[i] = float32(math.Log(float64(count)))
		}
		inv := 1 / count
		mean, std := row[m.logCounts:m.logCounts+dim], row[m.logCounts+dim:]
		for i := range dim {
			mu := j.sum[i] * inv
			v := j.sumsq[i]*inv - float32(mu*mu)
			mean[i] = mu
			std[i] = float32(math.Sqrt(float64(max(v, m.varFloor))))
		}
	}
	y := make([]float32, len(jobs)*xvecDim)
	out := make([][]float32, len(jobs))
	for k := range jobs {
		out[k] = y[k*xvecDim : (k+1)*xvecDim : (k+1)*xvecDim]
		copy(out[k], m.tdnn6.bias)
	}
	blas.Sgemm(len(jobs), xvecDim, in, x, m.tdnn6.w, 1, y)
	// nnet3-xvector-compute averages its chunks weighted by size, AddVec(n, v)
	// then Scale(1/n), even for one chunk. In float32 that round trip is not
	// always the identity, so it is done here too, the way Kaldi does it.
	for k, j := range jobs {
		w := float32(j.n)
		scale := float32(1 / float64(w))
		for i, v := range out[k] {
			out[k][i] = float32(w*v) * scale
		}
	}
	return out
}

// scratch keeps the per-layer buffers of frameLevel between blocks.
type scratch struct{ bufs [][]float32 }

// buf returns buffer i with length n, growing it when it is too small.
func (s *scratch) buf(i, n int) []float32 {
	for len(s.bufs) <= i {
		s.bufs = append(s.bufs, nil)
	}
	if cap(s.bufs[i]) < n {
		s.bufs[i] = make([]float32, n)
	}
	return s.bufs[i][:n]
}

// frameLevel runs tdnn1..tdnn5 on n input rows and returns the
// n-ctxL-ctxR tdnn5 batchnorm rows; row r belongs to input row r+ctxL.
func (m *Model) frameLevel(in []float32, n int, s *scratch) []float32 {
	x, dim := in, featDim
	for li := range m.layers {
		l := &m.layers[li]
		lo, hi := l.offsets[0], l.offsets[len(l.offsets)-1]
		rows := n - (hi - lo)
		y := s.buf(2*li+1, rows*l.out)
		parallelRows(rows, func(r0, r1 int) {
			for r := r0; r < r1; r++ {
				copy(y[r*l.out:(r+1)*l.out], l.bias)
			}
		})
		if li == 0 {
			// tdnn1's 30-wide rows are spliced side by side for one GEMM.
			spliced := s.buf(2*li, rows*l.in)
			parallelRows(rows, func(r0, r1 int) {
				for r := r0; r < r1; r++ {
					for k, o := range l.offsets {
						src := (r + o - lo) * dim
						copy(spliced[r*l.in+k*dim:r*l.in+(k+1)*dim], x[src:src+dim])
					}
				}
			})
			blas.Sgemm(rows, l.out, l.in, spliced, l.w, 1, y)
		} else {
			// One GEMM per offset straight from the input rows. The GEMM's
			// block depth divides 512, so this adds the same sums in the same
			// order as one GEMM over spliced rows (blas TestGemmSplitK).
			for k, o := range l.offsets {
				blas.Sgemm(rows, l.out, dim, x[(o-lo)*dim:], l.w[k*dim*l.out:], 1, y)
			}
		}
		parallelRows(rows, func(r0, r1 int) {
			for r := r0; r < r1; r++ {
				row := y[r*l.out : (r+1)*l.out]
				for i, v := range row {
					row[i] = float32(max(v, 0)*l.scale[i]) + l.offset[i]
				}
			}
		})
		x, dim, n = y, l.out, rows
	}
	return x
}

// parallelRows splits [0, n) into contiguous parts, one per CPU.
func parallelRows(n int, fn func(r0, r1 int)) {
	parts := min(runtime.GOMAXPROCS(0), max(n/64, 1))
	par.For(parts, func(p int) { fn(p*n/parts, (p+1)*n/parts) })
}
