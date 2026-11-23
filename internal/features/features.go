// Package features computes the input of the x-vector network the way the
// BBC speech segmenter recipe does with Kaldi: compute-mfcc-feats with
// conf/mfcc.conf, then apply-cmvn-sliding --norm-vars=false --center=true
// --cmn-window=300.
//
// Ported from Kaldi feat/feature-window.cc, feat/feature-mfcc.cc,
// feat/mel-computations.cc, feat/feature-functions.cc and
// matrix/matrix-functions.cc. Kaldi works in float32 inside a frame; this
// port uses float64 there, a difference far smaller than the dither noise.
package features

import (
	"math"
	"math/rand/v2"
	"sync"

	"gonum.org/v1/gonum/dsp/fourier"

	"github.com/oszuidwest/zwfm-speech-segments/internal/par"
)

// numCeps is the number of cepstra per frame (--num-ceps=30).
const numCeps = 30

// Options from conf/mfcc.conf and apply-cmvn-sliding, plus Kaldi defaults.
const (
	frameShift  = 160 // 10 ms at 16 kHz
	frameLength = 400 // 25 ms at 16 kHz
	paddedLen   = 512 // round-to-power-of-two
	numFFTBins  = paddedLen / 2
	numMelBins  = 30
	sampleFreq  = 16000.0
	lowFreq     = 20.0
	highFreq    = 7600.0
	preemph     = 0.97
	dither      = 1.0
	lifter      = 22.0
	cmnWindow   = 300
	ditherSeed  = 0x7a77_5345_4746_0001
	chunk       = 1024 // frames a goroutine takes at a time
)

// fltEpsilon is std::numeric_limits<float>::epsilon(), Kaldi's floor for
// energies before the log.
const fltEpsilon = 1.1920928955078125e-07

// melBin is one triangular mel filter: weights apply to the FFT power bins
// starting at offset.
type melBin struct {
	offset  int
	weights []float64
}

// tables holds everything that depends only on the options.
type tables struct {
	window [frameLength]float64
	mel    [numMelBins]melBin
	dct    [numCeps][numMelBins]float64 // DCT rows with the lifter folded in
}

// getTables builds the tables once, on first use.
var getTables = sync.OnceValue(newTables)

// newTables computes the window, the mel filter bank and the liftered DCT
// matrix.
func newTables() *tables {
	t := &tables{}
	// FeatureWindowFunction, window_type=povey.
	a := 2 * math.Pi / float64(frameLength-1)
	for i := range t.window {
		t.window[i] = math.Pow(0.5-0.5*math.Cos(a*float64(i)), 0.85)
	}

	// MelBanks with vtln_warp=1 and htk_mode=false.
	mel := func(f float64) float64 { return 1127.0 * math.Log(1.0+f/700.0) }
	melLow, melHigh := mel(lowFreq), mel(highFreq)
	delta := (melHigh - melLow) / (numMelBins + 1)
	binWidth := sampleFreq / paddedLen
	for b := range t.mel {
		left := melLow + float64(b)*delta
		center := melLow + float64(b+1)*delta
		right := melLow + float64(b+2)*delta
		first := -1
		var w []float64
		for i := range numFFTBins {
			m := mel(binWidth * float64(i))
			if m <= left || m >= right {
				continue
			}
			if first == -1 { // mel is monotonic, so the run is contiguous
				first = i
			}
			if m <= center {
				w = append(w, (m-left)/(center-left))
			} else {
				w = append(w, (right-m)/(right-center))
			}
		}
		t.mel[b] = melBin{offset: first, weights: w}
	}

	// ComputeDctMatrix (num_bins x num_bins, first num_ceps rows) times
	// ComputeLifterCoeffs.
	for k := range numCeps {
		norm := math.Sqrt(2.0 / numMelBins)
		if k == 0 {
			norm = math.Sqrt(1.0 / numMelBins)
		}
		lift := 1.0 + 0.5*lifter*math.Sin(math.Pi*float64(k)/lifter)
		for n := range numMelBins {
			t.dct[k][n] = lift * norm * math.Cos(math.Pi/numMelBins*(float64(n)+0.5)*float64(k))
		}
	}

	return t
}

// numFrames is Kaldi's NumFrames with snip-edges=false and flush=true.
func numFrames(numSamples int) int {
	return (numSamples + frameShift/2) / frameShift
}

// worker holds the scratch space for one chunk of frames.
type worker struct {
	t     *tables
	pcg   *rand.PCG
	rng   *rand.Rand
	fft   *fourier.FFT       // not safe for concurrent use
	frame [paddedLen]float64 // samples past frameLength stay zero
	coeff []complex128
	power [numFFTBins]float64
	logE  [numMelBins]float64
}

// newWorker returns a worker with its own FFT plan and dither generator.
func newWorker(t *tables) *worker {
	pcg := rand.NewPCG(0, 0)
	return &worker{
		t: t, pcg: pcg, rng: rand.New(pcg), //nolint:gosec // dither noise must be reproducible, not unpredictable
		fft: fourier.NewFFT(paddedLen), coeff: make([]complex128, paddedLen/2+1),
	}
}

// compute writes the MFCC of frame f into out (len numCeps). It follows
// ExtractWindow, ProcessWindow and MfccComputer::Compute.
func (w *worker) compute(pcm []int16, f int, out []float32) {
	t, x, n := w.t, w.frame[:frameLength], len(pcm)

	// Kaldi draws its dither from rand_r, which cannot be reproduced. This
	// uses Gaussian noise of the same scale from a generator seeded by the
	// frame index, so the output does not depend on how frames are split
	// over goroutines.
	w.pcg.Seed(ditherSeed, uint64(f)*0x9e3779b97f4a7c15) //nolint:gosec // frame index is never negative

	// FirstSampleOfFrame with snip-edges=false; samples outside the
	// signal are reflected back into it.
	start := f*frameShift + frameShift/2 - frameLength/2
	sum := 0.0
	for s := range x {
		i := start + s
		for i < 0 || i >= n {
			if i < 0 {
				i = -i - 1
			} else {
				i = 2*n - 1 - i
			}
		}
		x[s] = float64(pcm[i]) + dither*w.rng.NormFloat64()
		sum += x[s]
	}

	// remove-dc-offset, then the raw log energy before preemphasis.
	mean, energy := sum/frameLength, 0.0
	for s := range x {
		x[s] -= mean
		energy += x[s] * x[s]
	}
	logEnergy := math.Log(max(energy, fltEpsilon))

	for s := frameLength - 1; s > 0; s-- {
		x[s] -= preemph * x[s-1]
	}
	x[0] -= preemph * x[0]

	for s := range x {
		x[s] *= t.window[s]
	}
	w.fft.Coefficients(w.coeff, w.frame[:])
	// Bin 256 (Nyquist) is left out: no mel bin reaches it below 7600 Hz.
	for k := range w.power {
		c := w.coeff[k]
		w.power[k] = real(c)*real(c) + imag(c)*imag(c)
	}

	for b := range t.mel {
		e := 0.0
		for i, wt := range t.mel[b].weights {
			e += wt * w.power[t.mel[b].offset+i]
		}
		w.logE[b] = math.Log(max(e, fltEpsilon))
	}

	out[0] = float32(logEnergy) // use-energy with raw-energy, energy-floor 0
	for k := 1; k < numCeps; k++ {
		c := 0.0
		for i, v := range w.logE {
			c += t.dct[k][i] * v
		}
		out[k] = float32(c)
	}
}

// mfcc returns the MFCC features of 16 kHz mono PCM, row-major,
// frames x numCeps, as compute-mfcc-feats --config=conf/mfcc.conf would.
func mfcc(pcm []int16) (frames int, feats []float32) {
	frames = numFrames(len(pcm))
	feats = make([]float32, frames*numCeps)
	t := getTables()
	// Goroutines take small chunks in turn, so slower efficiency cores do
	// not hold up the end.
	par.For((frames+chunk-1)/chunk, func(c int) {
		w := newWorker(t)
		for f := c * chunk; f < min((c+1)*chunk, frames); f++ {
			w.compute(pcm, f, feats[f*numCeps:(f+1)*numCeps])
		}
	})
	return frames, feats
}

// slidingCMN subtracts from each frame the mean over a centred window of
// 300 frames, shifted inward at the edges (SlidingWindowCmnInternal with
// center=true, norm-vars=false). Sums are kept in float64 like Kaldi.
func slidingCMN(frames int, feats []float32) []float32 {
	out := make([]float32, frames*numCeps)
	var sum [numCeps]float64
	addRow := func(r int, sign float64) {
		for d, v := range feats[r*numCeps : (r+1)*numCeps] {
			sum[d] += sign * float64(v)
		}
	}
	lastStart, lastEnd := -1, -1
	for t := range frames {
		start := t - cmnWindow/2
		end := start + cmnWindow
		if start < 0 {
			end -= start
			start = 0
		}
		if end > frames {
			start = max(start-(end-frames), 0)
			end = frames
		}
		if lastStart == -1 {
			for r := start; r < end; r++ {
				addRow(r, 1)
			}
		} else {
			if start > lastStart {
				addRow(lastStart, -1)
			}
			if end > lastEnd {
				addRow(lastEnd, 1)
			}
		}
		lastStart, lastEnd = start, end
		scale := -1.0 / float64(end-start)
		row := feats[t*numCeps : (t+1)*numCeps]
		for d, v := range row {
			out[t*numCeps+d] = float32(float64(v) + scale*sum[d])
		}
	}
	return out
}

// Compute returns the CMN-normalised MFCC features of 16 kHz mono PCM,
// row-major, frames x numCeps. This is what the x-vector network consumes.
func Compute(pcm []int16) (frames int, feats []float32) {
	frames, raw := mfcc(pcm)
	compress(frames, raw) // the recipe stores MFCC and CMN features compressed
	feats = slidingCMN(frames, raw)
	compress(frames, feats)
	return frames, feats
}
