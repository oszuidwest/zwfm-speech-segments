// Package diarize ports stages 5 to 10 of the BBC speech segmenter recipe
// (run-segmentation.sh with its defaults) to Go: x-vector VAD, PLDA
// scoring, agglomerative clustering and the RTTM/STM post-processing.
package diarize

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"

	"github.com/oszuidwest/zwfm-speech-segments/internal/kaldi"
	"github.com/oszuidwest/zwfm-speech-segments/internal/xvector"
)

// Recipe defaults.
const (
	vadThreshold     = 0.2  // --vad-threshold
	clusterThreshold = -0.3 // --cluster-threshold
	targetEnergy     = 0.9  // score-plda.sh --target-energy

	// maxWindows is the most x-vectors Run accepts. Above it Kaldi's
	// agglomerative-cluster switches to a two-pass method that is not ported,
	// and the n×n score matrix grows past 4 GB.
	maxWindows = 32767
)

// Window is the time span of one x-vector, in seconds.
type Window = xvector.Window

// Segment is one speech turn, one line of diarize.stm. Start and End are in
// seconds from the start of the recording.
type Segment struct {
	Start, End float64
	Speaker    int // cluster label; the recipe prints it as audio_S%05d
}

// Model holds the PLDA back end and the VAD classifier.
type Model struct {
	vad  []vadFold
	mean []float64 // mean.vec
	lda  kaldi.Matrix
	plda plda
}

// vadFold is one fold of the calibrated classifier: the LinearSVC weights
// and intercept, and the sigmoid calibration parameters A and B.
type vadFold struct {
	Coef      []float64 `json:"coef"`
	Intercept float64   `json:"intercept"`
	A         float64   `json:"a"`
	B         float64   `json:"b"`
}

// vadJSON is the VAD classifier exported from the BBC model.
//
//go:embed vad-classifier.json
var vadJSON []byte

// Load reads plda, transform.mat and mean.vec from the recipe's model
// directory and checks their dimensions against the embedded VAD classifier.
func Load(modelDir string) (*Model, error) {
	var vad struct {
		Classes []string  `json:"classes"`
		Folds   []vadFold `json:"folds"`
	}
	if err := json.Unmarshal(vadJSON, &vad); err != nil {
		return nil, fmt.Errorf("vad classifier: %w", err)
	}
	if len(vad.Classes) != 2 || vad.Classes[1] != "speech" || len(vad.Folds) == 0 {
		return nil, fmt.Errorf("vad classifier: unexpected classes %v", vad.Classes)
	}
	m := &Model{vad: vad.Folds}
	var err error
	if m.mean, err = kaldi.ReadVector(filepath.Join(modelDir, "mean.vec")); err != nil {
		return nil, err
	}
	if m.lda, err = kaldi.ReadMatrix(filepath.Join(modelDir, "transform.mat")); err != nil {
		return nil, err
	}
	if m.lda.Cols != len(m.mean) {
		return nil, fmt.Errorf("transform.mat has %d columns for dim %d", m.lda.Cols, len(m.mean))
	}
	if m.plda, err = readPlda(filepath.Join(modelDir, "plda")); err != nil {
		return nil, err
	}
	if m.plda.dim() != m.lda.Rows {
		return nil, fmt.Errorf("plda dim %d, transform.mat rows %d", m.plda.dim(), m.lda.Rows)
	}
	for _, f := range m.vad {
		if len(f.Coef) != len(m.mean) {
			return nil, fmt.Errorf("vad classifier dim %d, x-vector dim %d", len(f.Coef), len(m.mean))
		}
	}
	return m, nil
}

// Run does VAD, PLDA scoring, clustering and the post-processing for one
// recording. As in the recipe, scoring and clustering use all x-vectors;
// only the RTTM is built from the windows that pass the VAD. Run returns an
// error for more than maxWindows x-vectors and panics if wins and xvectors
// differ in length.
func (m *Model) Run(wins []Window, xvectors [][]float32) ([]Segment, error) {
	if len(wins) != len(xvectors) {
		panic(fmt.Sprintf("diarize: %d windows for %d x-vectors", len(wins), len(xvectors)))
	}
	if len(xvectors) > maxWindows {
		return nil, fmt.Errorf("diarize: %d windows, at most %d (about %.1f hours of audio) are supported",
			len(xvectors), maxWindows, float64(maxWindows)*0.75/3600)
	}
	if len(xvectors) == 0 {
		return nil, nil
	}
	// filter_unvoiced_xvectors.py: keep windows with speech >= threshold.
	keep := make([]bool, len(xvectors))
	for i, p := range m.speechProbs(xvectors) {
		keep[i] = p >= vadThreshold
	}
	return postProcess(wins, keep, cluster(m.scores(xvectors), len(xvectors), clusterThreshold)), nil
}

// postProcess is make_rttm.py, merge_single_xvector_clusters.py,
// pad_segments.py and convert_rttm_to_stm.py.
func postProcess(wins []Window, keep []bool, labels []int) []Segment {
	turns := padSegments(mergeSingle(makeRTTM(wins, keep, labels)))
	segs := make([]Segment, len(turns))
	for i, t := range turns {
		segs[i] = Segment{t.Beg, t.Beg + t.Dur, t.Label}
	}
	return segs
}

// speechProbs is xvector_utils.py predict: the "speech" column of sklearn's
// CalibratedClassifierCV(sigmoid).predict_proba over
// Pipeline(Normalizer(l1), LinearSVC), averaged over the folds.
func (m *Model) speechProbs(xvectors [][]float32) []float64 {
	out := make([]float64, len(xvectors))
	xn := make([]float32, len(m.mean))
	for i, x := range xvectors {
		// Normalizer(norm="l1") runs in float32 with numpy's pairwise sum.
		abs := make([]float32, len(x))
		for k, v := range x {
			abs[k] = float32(math.Abs(float64(v)))
		}
		norm := pairwiseSum(abs)
		if norm == 0 {
			norm = 1
		}
		for k, v := range x {
			xn[k] = v / norm
		}
		var mean float64
		for _, f := range m.vad {
			// The explicit float64 conversions round each product before it
			// is added, which stops the Go compiler from fusing the two into
			// an FMA and changing the rounding.
			var df float64
			for k, c := range f.Coef {
				df += float64(float64(xn[k]) * c)
			}
			df += f.Intercept
			mean += 1 / (1 + math.Exp(float64(f.A*df)+f.B))
		}
		out[i] = mean / float64(len(m.vad))
	}
	return out
}

// pairwiseSum is numpy's pairwise_sum for float32 (PW_BLOCKSIZE 128).
func pairwiseSum(a []float32) float32 {
	n := len(a)
	switch {
	case n < 8:
		var res float32
		for _, v := range a {
			res += v
		}
		return res
	case n <= 128:
		var r [8]float32
		copy(r[:], a[:8])
		i := 8
		for ; i < n-n%8; i += 8 {
			for j := range r {
				r[j] += a[i+j]
			}
		}
		res := ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
		for ; i < n; i++ {
			res += a[i]
		}
		return res
	}
	n2 := n / 2
	n2 -= n2 % 8
	return pairwiseSum(a[:n2]) + pairwiseSum(a[n2:])
}
