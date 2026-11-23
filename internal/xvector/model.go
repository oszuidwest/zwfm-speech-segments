// Package xvector computes the BBC speech segmenter's x-vectors natively: the
// sliding windows of the recipe's extract-xvectors.sh and the tdnn6.affine
// output of model/final.raw as nnet3-xvector-compute produces it.
package xvector

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/oszuidwest/zwfm-speech-segments/internal/kaldi"
)

// xvecDim is the size of one x-vector (the tdnn6.affine output).
const xvecDim = 512

// featDim is the size of one input feature frame.
const featDim = 30

// affine is y = x·w + bias with w stored in×out, the transpose of Kaldi's
// LinearParams, so rows of frames multiply it directly.
type affine struct {
	in, out int
	w, bias []float32
}

// layer is one frame-level TDNN layer: Append of offsets, affine, ReLU and
// batchnorm in test mode (y = x·scale + offset).
type layer struct {
	offsets []int
	affine
	scale, offset []float32
}

// Model is the part of final.raw up to tdnn6.affine (model/extract.config).
type Model struct {
	layers             []layer
	ctxL, ctxR         int // input frames a tdnn5 frame needs before and after it
	poolL, poolR       int // stats-pooling context around output t=0
	logCounts          int // log-count features in front of mean and stddev
	varFloor           float32
	tdnn6              affine
	minChunk, maxChunk int // window sizes in frames that Kaldi runs as one chunk
}

// tdnnOffsets are the Append offsets of final.raw's frame-level layers, as
// its config lines and nnet3-info list them; Load checks the dimensions
// against them.
var tdnnOffsets = [][]int{{-2, -1, 0, 1, 2}, {-2, 0, 2}, {-3, 0, 3}, {0}, {0}}

// Load reads final.raw, min_chunk_size and max_chunk_size from modelDir.
func Load(modelDir string) (*Model, error) {
	m := &Model{}
	var err error
	if m.minChunk, err = readInt(filepath.Join(modelDir, "min_chunk_size")); err != nil {
		return nil, err
	}
	if m.maxChunk, err = readInt(filepath.Join(modelDir, "max_chunk_size")); err != nil {
		return nil, err
	}
	comps, err := readNnet(filepath.Join(modelDir, "final.raw"), "tdnn6.affine")
	if err != nil {
		return nil, err
	}
	if err := m.loadLayers(comps); err != nil {
		return nil, err
	}
	dim := m.layers[len(m.layers)-1].out
	if err := m.loadPooling(comps, dim); err != nil {
		return nil, err
	}
	if m.tdnn6, err = toAffine(comps["tdnn6.affine"]); err != nil {
		return nil, fmt.Errorf("tdnn6.affine: %w", err)
	}
	if m.tdnn6.in != m.logCounts+2*dim || m.tdnn6.out != xvecDim {
		return nil, fmt.Errorf("tdnn6.affine: %dx%d", m.tdnn6.in, m.tdnn6.out)
	}
	if m.minChunk < m.ctxL+m.ctxR+1 || m.maxChunk < m.minChunk || m.poolR < m.ctxL {
		return nil, fmt.Errorf("model: chunk sizes %d/%d or pooling context unusable",
			m.minChunk, m.maxChunk)
	}
	return m, nil
}

// loadLayers reads tdnn1..tdnn5 and sums their context into ctxL and ctxR.
func (m *Model) loadLayers(comps map[string]fields) error {
	dim := featDim
	for i, offs := range tdnnOffsets {
		name := fmt.Sprintf("tdnn%d", i+1)
		l := layer{offsets: offs}
		var err error
		if l.affine, err = toAffine(comps[name+".affine"]); err != nil {
			return fmt.Errorf("%s.affine: %w", name, err)
		}
		if l.in != len(offs)*dim {
			return fmt.Errorf("%s.affine: input dim %d, want %d", name, l.in, len(offs)*dim)
		}
		if _, ok := comps[name+".relu"]; !ok {
			return fmt.Errorf("%s.relu missing", name)
		}
		if l.scale, l.offset, err = toBatchnorm(comps[name+".batchnorm"], l.out); err != nil {
			return fmt.Errorf("%s.batchnorm: %w", name, err)
		}
		m.ctxL -= offs[0]
		m.ctxR += offs[len(offs)-1]
		m.layers = append(m.layers, l)
		dim = l.out
	}
	return nil
}

// loadPooling checks the stats components on top of the dim-wide tdnn5
// output and reads the pooling context, log-count size and variance floor.
func (m *Model) loadPooling(comps map[string]fields, dim int) error {
	ext, pool := comps["stats-extraction-0-10000"], comps["stats-pooling-0-10000"]
	if ext == nil || pool == nil {
		return fmt.Errorf("final.raw: stats components missing")
	}
	// Output period 1 makes each extraction row one frame's (1, x, x²), so
	// pooling sums frames one by one; other periods are not handled.
	// <IncludeVarinance> is spelled the way Kaldi writes it.
	if ext.int("<InputDim>") != dim || ext.int("<InputPeriod>") != 1 ||
		ext.int("<OutputPeriod>") != 1 || ext["<IncludeVarinance>"] != true {
		return fmt.Errorf("stats-extraction: unsupported config %v", ext)
	}
	if pool.int("<InputDim>") != 2*dim+1 || pool.int("<InputPeriod>") != 1 ||
		pool["<OutputStddevs>"] != true {
		return fmt.Errorf("stats-pooling: unsupported config %v", pool)
	}
	m.poolL, m.poolR = pool.int("<LeftContext>"), pool.int("<RightContext>")
	m.logCounts = pool.int("<NumLogCountFeatures>")
	m.varFloor = float32(pool.float("<VarianceFloor>"))
	return nil
}

// readInt reads a file that holds one decimal integer.
func readInt(path string) (int, error) {
	b, err := os.ReadFile(path) //nolint:gosec // G304: reading the model directory the caller chose is the purpose
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(string(b)))
}

// fields holds one component's "<Tag> value" pairs: float64, int, bool,
// []float64 or kaldi.Matrix.
type fields map[string]any

// int returns field k as an int, or 0 when it is missing or not an int.
func (f fields) int(k string) int {
	v, _ := f[k].(int)
	return v
}

// float returns field k as a float64, or 0 when it is missing or not a float.
func (f fields) float(k string) float64 {
	v, _ := f[k].(float64)
	return v
}

// intTags are the 4-byte fields Kaldi writes as int32; other 4- and 8-byte
// fields are floats.
var intTags = map[string]bool{"<Dim>": true, "<BlockDim>": true, "<InputDim>": true,
	"<InputPeriod>": true, "<OutputPeriod>": true, "<LeftContext>": true,
	"<RightContext>": true, "<NumLogCountFeatures>": true}

// readNnet reads an nnet3 raw model (Nnet::Read in nnet3/nnet-nnet.cc): the
// text config lines, then "<NumComponents>" and each component, here up to
// and including the component named last.
func readNnet(path, last string) (map[string]fields, error) {
	r, f, err := kaldi.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }() // read-only file: a close error loses nothing
	if err := r.Expect("<Nnet3>"); err != nil {
		return nil, err
	}
	if err := r.SkipTo("<NumComponents>"); err != nil {
		return nil, err
	}
	n, err := r.Int32()
	if err != nil {
		return nil, err
	}
	comps := map[string]fields{}
	for range n {
		if err := r.Expect("<ComponentName>"); err != nil {
			return nil, err
		}
		name, err := r.Token()
		if err != nil {
			return nil, err
		}
		typ, err := r.Token()
		if err != nil {
			return nil, err
		}
		if comps[name], err = readFields(r, "</"+typ[1:]); err != nil {
			return nil, fmt.Errorf("%s %s: %w", name, typ, err)
		}
		if name == last {
			return comps, nil
		}
	}
	return nil, fmt.Errorf("%s: component %s not found", path, last)
}

// readFields reads tagged values until the closing tag end, telling the value
// types apart by their first bytes.
func readFields(r *kaldi.Reader, end string) (fields, error) {
	f := fields{}
	for {
		tag, err := r.Token()
		if err != nil {
			return nil, err
		}
		if tag == end {
			return f, nil
		}
		p, err := r.Peek(2)
		if err != nil {
			return nil, err
		}
		var v any
		switch {
		case p[0] == '<': // a tag without a value
			continue
		case (p[0] == 'F' || p[0] == 'D') && p[1] == 'M':
			v, err = r.Matrix()
		case (p[0] == 'F' || p[0] == 'D') && p[1] == 'V':
			v, err = r.Vector()
		case p[0] == 'T' || p[0] == 'F':
			v, err = r.Bool()
		case intTags[tag]:
			var i int32
			i, err = r.Int32()
			v = int(i)
		default:
			v, err = r.Float()
		}
		if err != nil {
			return nil, fmt.Errorf("%s: %w", tag, err)
		}
		f[tag] = v
	}
}

// toAffine converts an affine component to float32, transposing LinearParams
// to in×out.
func toAffine(f fields) (affine, error) {
	lin, ok1 := f["<LinearParams>"].(kaldi.Matrix)
	bias, ok2 := f["<BiasParams>"].([]float64)
	if !ok1 || !ok2 || len(bias) != lin.Rows {
		return affine{}, fmt.Errorf("missing or inconsistent parameters")
	}
	a := affine{in: lin.Cols, out: lin.Rows, w: make([]float32, lin.Rows*lin.Cols),
		bias: make([]float32, lin.Rows)}
	for o := range lin.Rows {
		for i, v := range lin.Row(o) {
			a.w[i*a.out+o] = float32(v)
		}
		a.bias[o] = float32(bias[o])
	}
	return a, nil
}

// toBatchnorm derives the test-mode scale and offset in float32 like
// BatchNormComponent::Read and ComputeDerived (nnet-normalize-component.cc).
func toBatchnorm(f fields, dim int) (scale, offset []float32, err error) {
	mean, ok1 := f["<StatsMean>"].([]float64)
	vari, ok2 := f["<StatsVar>"].([]float64)
	if !ok1 || !ok2 || f.int("<Dim>") != dim || f.int("<BlockDim>") != dim ||
		len(mean) != dim || len(vari) != dim {
		return nil, nil, fmt.Errorf("missing stats or block-dim != dim")
	}
	count, eps := float32(f.float("<Count>")), float32(f.float("<Epsilon>"))
	rms := float32(f.float("<TargetRms>"))
	if count <= 0 || eps <= 0 || rms <= 0 {
		return nil, nil, fmt.Errorf("count %v, epsilon %v, target-rms %v", count, eps, rms)
	}
	scale, offset = make([]float32, dim), make([]float32, dim)
	inv := float32(1 / float64(count))
	for i := range dim {
		m, v := float32(mean[i]), float32(vari[i])
		sum := m * count
		sumsq := float32(v+float32(m*m)) * count
		off := sum * float32(-1/float64(count))
		s := float32(sumsq*inv) - float32(off*off)
		s = max(s, 0) + eps
		s = float32(math.Pow(float64(s), -0.5)) * rms
		scale[i], offset[i] = s, off*s
	}
	return scale, offset, nil
}
