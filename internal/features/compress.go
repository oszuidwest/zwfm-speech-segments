package features

import (
	"math"
	"slices"
	"sync"
)

// compress rounds features the way `copy-feats --compress=true` stores them:
// Kaldi's CompressedMatrix with method kSpeechFeature
// (matrix/compressed-matrix.cc), one byte per value between four per-column
// percentiles. The recipe keeps both the MFCC and the CMN features
// compressed, and the speaker clustering is sensitive to this rounding, so
// it is reproduced exactly, including where C++ computes in float and where
// in double. The explicit float32 conversions also stop Go from fusing
// a*b+c into one FMA instruction. data is frames x numCeps and is rounded
// in place.
func compress(frames int, data []float32) {
	if frames <= 8 {
		return // Kaldi uses a two-byte format here; real audio is never this short
	}
	lo, hi := slices.Min(data), slices.Max(data)
	if hi == lo {
		hi = float32(float64(lo) + (1.0 + math.Abs(float64(lo))))
	}
	g := header{min: lo, rng: hi - lo}

	var wg sync.WaitGroup
	for c := range numCeps {
		wg.Go(func() {
			col := make([]float32, frames)
			for r := range frames {
				col[r] = data[r*numCeps+c]
			}
			v := quartiles(col)
			u0 := min(g.toUint16(v[0]), 65532)
			u25 := min(max(g.toUint16(v[1]), u0+1), 65533)
			u75 := min(max(g.toUint16(v[2]), u25+1), 65534)
			u100 := max(g.toUint16(v[3]), u75+1)
			p := [4]float32{g.toFloat(u0), g.toFloat(u25), g.toFloat(u75), g.toFloat(u100)}
			for r := range frames {
				i := r*numCeps + c
				data[i] = fromChar(p, toChar(p, data[i]))
			}
		})
	}
	wg.Wait()
}

// quartiles returns the values at sorted positions 0, n/4, 3*(n/4) and n-1
// of col, the four percentiles of Kaldi's ComputeColHeader. It reorders col.
func quartiles(col []float32) [4]float32 {
	q := len(col) / 4
	return [4]float32{slices.Min(col), nth(col, q), nth(col[q:], 2*q), slices.Max(col)}
}

// nth returns the value at position k of s in sorted order, like the
// std::nth_element that Kaldi uses. It reorders s so that no value before k
// is larger and none after k is smaller, so a later position can be selected
// in s[k:]. s must not hold NaN.
func nth(s []float32, k int) float32 {
	for len(s) > 1 {
		p := s[len(s)/2]
		i, j := 0, len(s)-1
		for i <= j {
			for s[i] < p {
				i++
			}
			for s[j] > p {
				j--
			}
			if i <= j {
				s[i], s[j] = s[j], s[i]
				i, j = i+1, j-1
			}
		}
		switch {
		case k <= j:
			s = s[:j+1]
		case k >= i:
			s, k = s[i:], k-i
		default:
			return s[k] // s[j+1:i] all equal p
		}
	}
	return s[0]
}

// header is the GlobalHeader of a CompressedMatrix: the minimum and range of
// the whole matrix, which the per-column percentiles are quantised against.
type header struct{ min, rng float32 }

// toUint16 is FloatToUint16: float division, then `f * 65535 + 0.499` with
// the product in float and the sum in double.
func (g header) toUint16(v float32) uint16 {
	f := (v - g.min) / g.rng
	f = min(max(f, 0), 1)
	return uint16(int(float64(float32(f*65535)) + 0.499)) //nolint:gosec // f is clamped to [0, 1], so the value fits in uint16
}

// toFloat is Uint16ToFloat: min + range * 1.52590218966964e-05F * value, in float.
func (g header) toFloat(u uint16) float32 {
	return g.min + float32(float32(g.rng*1.52590218966964e-05)*float32(u))
}

// toChar is FloatToChar: the scaled value in float, + 0.5 in double.
func toChar(p [4]float32, v float32) uint8 {
	var ans int
	switch {
	case v < p[1]:
		f := (v - p[0]) / (p[1] - p[0])
		ans = min(max(int(float64(float32(f*64))+0.5), 0), 64)
	case v < p[2]:
		f := (v - p[1]) / (p[2] - p[1])
		ans = min(max(64+int(float64(float32(f*128))+0.5), 64), 192)
	default:
		f := (v - p[2]) / (p[3] - p[2])
		ans = min(max(192+int(float64(float32(f*63))+0.5), 192), 255)
	}
	return uint8(ans)
}

// fromChar is CharToFloat: (pB - pA) * value in float, then the scale
// (1/64, 1/128 or 1/63) and + pA in double, then rounded to float.
func fromChar(p [4]float32, b uint8) float32 {
	switch {
	case b <= 64:
		return float32(float64(p[0]) + float64(float64(float32((p[1]-p[0])*float32(b)))*(1/64.0)))
	case b <= 192:
		return float32(float64(p[1]) + float64(float64(float32((p[2]-p[1])*float32(int(b)-64)))*(1/128.0)))
	default:
		return float32(float64(p[2]) + float64(float64(float32((p[3]-p[2])*float32(int(b)-192)))*(1/63.0)))
	}
}
