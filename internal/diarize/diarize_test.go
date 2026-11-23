package diarize

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/oszuidwest/zwfm-speech-segments/internal/model"
)

// goldenDir holds the Kaldi reference output of the BBC recipe for one hour
// of radio (golden/ and seg/). Tests that need it are skipped when it is
// missing.
const goldenDir = "../../data/zuidwest/2026-11-23_08"

// golden is the reference data, indexed by x-vector key where it is keyed.
type golden struct {
	keys   []string
	wins   []Window
	xvecs  [][]float32
	probs  map[string]float64
	kept   map[string]bool
	labels map[string]string
	stm    string
}

// loadGolden reads the reference x-vectors, windows, VAD output, cluster
// labels and diarize.stm, or skips the test when they are missing.
func loadGolden(t *testing.T) *golden {
	t.Helper()
	if _, err := os.Stat(filepath.Join(goldenDir, "golden", "xvectors.txt")); err != nil {
		t.Skip("golden data missing")
	}
	g := &golden{probs: map[string]float64{}, kept: map[string]bool{}, labels: map[string]string{}}
	eachLine(t, "golden/xvectors.txt", func(f []string) {
		g.keys = append(g.keys, f[0])
		v := make([]float32, 0, len(f)-3)
		for _, s := range f[2 : len(f)-1] {
			x, err := strconv.ParseFloat(s, 32)
			if err != nil {
				t.Fatal(err)
			}
			v = append(v, float32(x))
		}
		g.xvecs = append(g.xvecs, v)
	})
	wins := map[string]Window{}
	eachLine(t, "seg/xvectors/subsegments", func(f []string) {
		wins[f[0]] = Window{Start: parse(t, f[2]), End: parse(t, f[3])}
	})
	for _, k := range g.keys {
		g.wins = append(g.wins, wins[k])
	}
	eachLine(t, "seg/xvectors/predictions.txt", func(f []string) {
		if f[0] != "xvector_key" {
			g.probs[f[0]] = parse(t, f[2])
		}
	})
	eachLine(t, "seg/xvectors/segments.filt", func(f []string) { g.kept[f[0]] = true })
	eachLine(t, "seg/clusters/labels", func(f []string) { g.labels[f[0]] = f[1] })
	stm, err := os.ReadFile(filepath.Join(goldenDir, "seg", "diarize.stm"))
	if err != nil {
		t.Fatal(err)
	}
	g.stm = string(stm)
	return g
}

// eachLine calls fn with the fields of each non-empty line of a golden file.
func eachLine(t *testing.T, name string, fn func([]string)) {
	t.Helper()
	f, err := os.Open(filepath.Join(goldenDir, name)) //nolint:gosec // G304: fixed test data paths
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 1<<24)
	for sc.Scan() {
		if fields := strings.Fields(sc.Text()); len(fields) > 0 {
			fn(fields)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
}

// parse parses a float64 or fails the test.
func parse(t *testing.T, s string) float64 {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// goldenScores reads golden/plda_scores.txt, an n x n Kaldi text matrix of
// the form "key [ values ]".
func goldenScores(t *testing.T, n int) []float32 {
	f, err := os.Open(filepath.Join(goldenDir, "golden", "plda_scores.txt"))
	if err != nil {
		t.Skip("plda_scores.txt missing")
	}
	defer func() { _ = f.Close() }()
	sc := bufio.NewScanner(f)
	sc.Split(bufio.ScanWords)
	sc.Scan() // key
	out := make([]float32, 0, n*n)
	for sc.Scan() {
		if w := sc.Text(); w != "[" && w != "]" {
			v, err := strconv.ParseFloat(w, 32)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, float32(v))
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(out) != n*n {
		t.Fatalf("plda_scores.txt has %d values, want %d", len(out), n*n)
	}
	return out
}

// sameClusters reports whether two labelings are equal up to renaming.
func sameClusters(a []int, keys []string, golden map[string]string) bool {
	fwd, back := map[int]string{}, map[string]int{}
	for i, l := range a {
		g := golden[keys[i]]
		if x, ok := fwd[l]; ok && x != g {
			return false
		}
		if x, ok := back[g]; ok && x != l {
			return false
		}
		fwd[l], back[g] = g, l
	}
	return true
}

// stmText formats segments as the recipe's diarize.stm.
func stmText(segs []Segment) string {
	var b strings.Builder
	for _, s := range segs {
		fmt.Fprintf(&b, "audio 0 audio_S%05d %.3f %.3f <speech>\n", s.Speaker, s.Start, s.End)
	}
	return b.String()
}

// TestGoldenBackEnd pins clustering and post-processing: from Kaldi's own
// scores and labels they must reproduce the recipe's labels and diarize.stm
// exactly.
func TestGoldenBackEnd(t *testing.T) {
	g := loadGolden(t)
	n := len(g.keys)
	labels := cluster(goldenScores(t, n), n, clusterThreshold)
	if !sameClusters(labels, g.keys, g.labels) {
		t.Error("cluster labels differ from seg/clusters/labels")
	}
	keep := make([]bool, n)
	gl := make([]int, n)
	for i, k := range g.keys {
		keep[i] = g.kept[k]
		gl[i], _ = strconv.Atoi(g.labels[k])
	}
	got := stmText(postProcess(g.wins, keep, gl))
	if got != g.stm {
		t.Errorf("diarize.stm from golden labels differs:\n%s", firstDiff(got, g.stm))
	}
}

// TestRun pins the whole back end on the golden x-vectors: the same VAD
// kept set, PLDA scores within 0.01, the same cluster labels and the same
// diarize.stm.
func TestRun(t *testing.T) {
	g := loadGolden(t)
	dir, err := model.Dir()
	if err != nil {
		t.Skip("model not available:", err)
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	n := len(g.keys)
	probs := m.speechProbs(g.xvecs)
	var maxDiff float64
	mismatch := 0
	for i, k := range g.keys {
		maxDiff = max(maxDiff, math.Abs(probs[i]-g.probs[k]))
		if (probs[i] >= vadThreshold) != g.kept[k] {
			mismatch++
		}
	}
	t.Logf("VAD: max |p - golden| = %.3g, kept-set mismatches %d of %d", maxDiff, mismatch, n)
	if mismatch > 0 {
		t.Error("VAD kept set differs from segments.filt")
	}

	scores := m.scores(g.xvecs)
	var maxAbs float64
	for i, v := range goldenScores(t, n) {
		maxAbs = max(maxAbs, math.Abs(float64(scores[i]-v)))
	}
	t.Logf("PLDA: max |score - golden| = %.3g", maxAbs)
	if maxAbs > 0.01 {
		t.Error("PLDA scores differ from golden/plda_scores.txt")
	}
	labels := cluster(scores, n, clusterThreshold)
	if !sameClusters(labels, g.keys, g.labels) {
		t.Error("cluster labels differ from seg/clusters/labels")
	}

	start := time.Now()
	segs := m.Run(g.wins, g.xvecs)
	t.Logf("Run: %v for %d x-vectors", time.Since(start), n)
	if got := stmText(segs); got != g.stm {
		t.Errorf("diarize.stm differs:\n%s", firstDiff(got, g.stm))
	}
}

// firstDiff describes the first line where a and b differ.
func firstDiff(a, b string) string {
	al, bl := strings.Split(a, "\n"), strings.Split(b, "\n")
	for i := range min(len(al), len(bl)) {
		if al[i] != bl[i] {
			return "line " + strconv.Itoa(i+1) + ":\n got  " + al[i] + "\n want " + bl[i]
		}
	}
	return "length " + strconv.Itoa(len(al)) + " vs " + strconv.Itoa(len(bl))
}

// TestLoad pins that the model files load and match the VAD classifier's
// dimension; it needs only the model files.
func TestLoad(t *testing.T) {
	dir, err := model.Dir()
	if err != nil {
		t.Skip("model not available:", err)
	}
	if _, err := Load(dir); err != nil {
		t.Fatal(err)
	}
}
