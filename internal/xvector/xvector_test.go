package xvector

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/oszuidwest/zwfm-speech-segments/internal/model"
)

// hourDir holds Kaldi's reference output for one hour of radio. The golden
// tests compare against it and skip when its files are missing.
var hourDir = filepath.Join("..", "..", "data", "zuidwest", "2026-11-23_08")

// need skips the test when any of paths is missing.
func need(t *testing.T, paths ...string) {
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("golden file missing: %s", p)
		}
	}
}

// lines calls fn for each line of the file at path.
func lines(t *testing.T, path string, fn func(string)) {
	f, err := os.Open(path) //nolint:gosec // G304: test reads its own golden files
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }() // read-only file: a close error loses nothing
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		fn(sc.Text())
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
}

// floats parses Kaldi text matrix fields, skipping the brackets.
func floats(t *testing.T, fields []string) []float32 {
	out := make([]float32, 0, len(fields))
	for _, f := range fields {
		if f == "[" || f == "]" {
			continue
		}
		v, err := strconv.ParseFloat(f, 32)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, float32(v))
	}
	return out
}

// readCMVN reads golden/cmvn.txt: "key  [", then one row of 30 per line.
func readCMVN(t *testing.T) (frames int, feats []float32) {
	lines(t, filepath.Join(hourDir, "golden", "cmvn.txt"), func(l string) {
		if strings.HasSuffix(l, "[") {
			return
		}
		feats = append(feats, floats(t, strings.Fields(l))...)
	})
	return len(feats) / featDim, feats
}

// TestWindowsGolden pins Windows to Kaldi's subsegments file, window by
// window.
func TestWindowsGolden(t *testing.T) {
	subs := filepath.Join(hourDir, "seg", "xvectors", "subsegments")
	need(t, subs)
	var want []string
	lines(t, subs, func(l string) {
		f := strings.Fields(l)
		want = append(want, f[2]+" "+f[3])
	})
	// The recording has 359999 frames (golden/cmvn.txt rows); its segment
	// ends at 3599.990.
	got := Windows(359999)
	if len(got) != len(want) {
		t.Fatalf("%d windows, Kaldi has %d", len(got), len(want))
	}
	for i, w := range got {
		if s := fmt.Sprintf("%.3f %.3f", w.Start, w.End); s != want[i] {
			t.Fatalf("window %d: %s, Kaldi %s", i, s, want[i])
		}
	}
}

// TestLoad checks that the installed model passes Load's dimension checks.
func TestLoad(t *testing.T) {
	dir, err := model.Dir()
	if err != nil {
		t.Skip("model not available:", err)
	}
	if _, err := Load(dir); err != nil {
		t.Fatal(err)
	}
}

// TestExtractGolden compares Extract with Kaldi's x-vectors for the hour and
// requires a minimum cosine similarity of 0.99999 for every window.
func TestExtractGolden(t *testing.T) {
	dir, err := model.Dir()
	if err != nil {
		t.Skip("model not available:", err)
	}
	golden := filepath.Join(hourDir, "golden", "xvectors.txt")
	need(t, golden, filepath.Join(hourDir, "golden", "cmvn.txt"))

	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	frames, feats := readCMVN(t)
	var want [][]float32
	lines(t, golden, func(l string) { want = append(want, floats(t, strings.Fields(l)[1:])) })

	got := m.Extract(frames, feats, Windows(frames))
	if len(got) != len(want) {
		t.Fatalf("%d x-vectors, Kaldi has %d", len(got), len(want))
	}

	minCos, sumCos, maxAbs, worst := 1.0, 0.0, 0.0, 0
	for i := range got {
		c, d := compare(got[i], want[i])
		sumCos += c
		maxAbs = max(maxAbs, d)
		if c < minCos {
			minCos, worst = c, i
		}
	}
	first, _ := compare(got[0], want[0])
	last, _ := compare(got[len(got)-1], want[len(want)-1])
	t.Logf("cosine min %.9f (window %d) mean %.9f, max |diff| %.3g, first %.9f, last %.9f",
		minCos, worst, sumCos/float64(len(got)), maxAbs, first, last)
	if minCos < 0.99999 {
		t.Errorf("min cosine %.9f below 0.99999", minCos)
	}
}

// compare returns the cosine similarity and largest absolute difference of a
// and b.
func compare(a, b []float32) (cos, maxAbs float64) {
	var ab, aa, bb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		ab, aa, bb = ab+x*y, aa+x*x, bb+y*y
		maxAbs = max(maxAbs, math.Abs(x-y))
	}
	return ab / math.Sqrt(aa*bb), maxAbs
}
