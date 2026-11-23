package features

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"
)

// goldenDir holds one hour of audio and the output of the Kaldi recipe on
// it, used as reference. Tests that need it skip when it is missing.
var goldenDir = filepath.Join("..", "..", "data", "zuidwest", "2026-11-23_08")

// readWAV returns the samples of a 16 kHz mono s16 WAV. It walks the RIFF
// chunks, because the file has a LIST chunk before "data".
func readWAV(t testing.TB, path string) []int16 {
	data, err := os.ReadFile(path) //nolint:gosec // path is a test fixture under goldenDir
	if err != nil {
		t.Skipf("no golden audio: %v", err)
	}
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		t.Fatalf("%s: not a WAV file", path)
	}
	for p := 12; p+8 <= len(data); {
		id, size := string(data[p:p+4]), int(binary.LittleEndian.Uint32(data[p+4:p+8]))
		body := data[p+8 : min(p+8+size, len(data))]
		switch id {
		case "fmt ":
			format, ch := binary.LittleEndian.Uint16(body), binary.LittleEndian.Uint16(body[2:])
			rate, bits := binary.LittleEndian.Uint32(body[4:]), binary.LittleEndian.Uint16(body[14:])
			if format != 1 || ch != 1 || rate != 16000 || bits != 16 {
				t.Fatalf("%s: want PCM mono 16 kHz s16, got fmt=%d ch=%d rate=%d bits=%d", path, format, ch, rate, bits)
			}
		case "data":
			pcm := make([]int16, len(body)/2)
			if _, err := binary.Decode(body, binary.LittleEndian, pcm); err != nil {
				t.Fatalf("%s: %v", path, err)
			}
			return pcm
		}
		p += 8 + size + size&1
	}
	t.Fatalf("%s: no data chunk", path)
	return nil
}

// readKaldiText reads a one-utterance Kaldi text archive ("key [", rows, "]").
func readKaldiText(t testing.TB, path string) (rows int, out []float32) {
	f, err := os.Open(path) //nolint:gosec // path is a test fixture under goldenDir
	if err != nil {
		t.Skipf("no golden features: %v", err)
	}
	defer func() { _ = f.Close() }()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Bytes()
		if bytes.HasSuffix(bytes.TrimSpace(line), []byte("[")) {
			continue
		}
		n := 0
		for field := range bytes.FieldsSeq(line) {
			if field[0] == ']' {
				continue
			}
			v, err := strconv.ParseFloat(string(field), 32)
			if err != nil {
				t.Fatalf("%s: %v", path, err)
			}
			out = append(out, float32(v))
			n++
		}
		if n != numCeps {
			t.Fatalf("%s: row %d has %d values", path, rows, n)
		}
		rows++
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return rows, out
}

// agreement returns the share of values equal to the golden value within the
// precision of Kaldi's text output, and the number that are a whole
// compression step or more apart.
func agreement(got, want []float32) (share float64, jumps int) {
	same := 0
	for i := range want {
		d := math.Abs(float64(got[i] - want[i]))
		if d <= 1e-5*max(1, math.Abs(float64(want[i]))) {
			same++
		} else if d > 1e-3 {
			jumps++
		}
	}
	return float64(same) / float64(len(want)), jumps
}

// TestCMNGolden pins slidingCMN and compress: fed Kaldi's own compressed
// MFCC, they must reproduce Kaldi's CMN features value for value. The few
// exceptions are values at a step edge, because the text file holds only
// 7 digits.
func TestCMNGolden(t *testing.T) {
	frames, kaldiMFCC := readKaldiText(t, filepath.Join(goldenDir, "golden", "mfcc.txt"))
	_, want := readKaldiText(t, filepath.Join(goldenDir, "golden", "cmvn.txt"))
	got := slidingCMN(frames, kaldiMFCC)
	compress(frames, got)
	share, jumps := agreement(got, want)
	t.Logf("%.5f%% equal, %d step jumps of %d values", 100*share, jumps, len(want))
	if share < 0.9999 {
		t.Errorf("CMN + compression differ from Kaldi")
	}
}

// TestComputeGolden pins the whole pipeline against Kaldi's CMN features.
// Kaldi's dither noise cannot be reproduced; it moves near-silent frames
// and, through the matrix extremes, the compression grid, so the test bounds
// the mean difference instead of requiring equality.
func TestComputeGolden(t *testing.T) {
	pcm := readWAV(t, filepath.Join(goldenDir, "audio.wav"))
	wantFrames, want := readKaldiText(t, filepath.Join(goldenDir, "golden", "cmvn.txt"))
	start := time.Now()
	frames, got := Compute(pcm)
	t.Logf("Compute took %v", time.Since(start))
	if frames != wantFrames {
		t.Fatalf("frames = %d, Kaldi has %d", frames, wantFrames)
	}
	sum, worst := 0.0, 0.0
	for i := range want {
		d := math.Abs(float64(got[i] - want[i]))
		sum, worst = sum+d, max(worst, d)
	}
	t.Logf("mean abs diff %.4f, max %.2f", sum/float64(len(want)), worst)
	if sum/float64(len(want)) > 0.02 {
		t.Errorf("features differ from Kaldi beyond dither")
	}
}

// TestQuartiles pins the selection in compress against a full sort, on
// columns with few distinct values and on random ones.
func TestQuartiles(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2)) //nolint:gosec // test data, not security
	for _, n := range []int{9, 10, 11, 12, 101, 1000} {
		for _, distinct := range []int{1, 2, 3, 0, -1} { // 0: random floats, -1: organ pipe
			col := make([]float32, n)
			for i := range col {
				switch {
				case distinct > 0:
					col[i] = float32(r.IntN(distinct))
				case distinct == 0:
					col[i] = r.Float32()
				default: // defeats the middle pivot; from about 15 values nth takes its sort fallback
					col[i] = float32(min(i, n-i))
				}
			}
			s, q := slices.Sorted(slices.Values(col)), n/4
			want := [4]float32{s[0], s[q], s[3*q], s[n-1]}
			if got := quartiles(col); got != want {
				t.Errorf("n=%d distinct=%d: got %v, want %v", n, distinct, got, want)
			}
		}
	}
}

// BenchmarkCompute times Compute on the golden hour of audio.
func BenchmarkCompute(b *testing.B) {
	pcm := readWAV(b, filepath.Join(goldenDir, "audio.wav"))
	for b.Loop() {
		Compute(pcm)
	}
}
