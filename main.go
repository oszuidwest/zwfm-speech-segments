// Command zwfm-speech-segments prints the speech segments and speakers in an
// audio recording as JSON.
//
// Usage:
//
//	zwfm-speech-segments AUDIO
//
// AUDIO is anything ffmpeg can read: a file, a URL, or - for standard input.
// For example:
//
//	zwfm-speech-segments recording.mp3
//	zwfm-speech-segments https://example.org/recording.wav
//	ffmpeg -i input.mkv -f wav - | zwfm-speech-segments -
//
// Each segment has a start and end time code, counted from the start of the
// recording, and a speaker number; speaker 1 is the voice with the most
// speaking time.
//
// It is a Go port of the BBC speech segmenter
// (github.com/bbc/bbc-speech-segmenter): Kaldi MFCC and sliding CMN, the
// x-vector network, the calibrated VAD classifier, PLDA scoring,
// agglomerative clustering and the recipe's post-processing.
//
// ffmpeg must be installed and on the PATH; it decodes the audio.
package main

import (
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"maps"
	"math"
	"os"
	"os/exec"
	"slices"
	"time"

	"github.com/oszuidwest/zwfm-speech-segments/internal/diarize"
	"github.com/oszuidwest/zwfm-speech-segments/internal/features"
	"github.com/oszuidwest/zwfm-speech-segments/internal/model"
	"github.com/oszuidwest/zwfm-speech-segments/internal/xvector"
)

// output is the JSON document written to stdout.
type output struct {
	Duration string    `json:"duration"`
	Segments []segment `json:"segments"`
}

// segment is one speech turn. Start and End are time codes from the start of
// the recording. Speaker 1 is the voice with the most speaking time; the
// model does not know who it is.
type segment struct {
	Start   string `json:"start"`
	End     string `json:"end"`
	Speaker int    `json:"speaker"`
}

func main() {
	log.SetFlags(0)
	if len(os.Args) != 2 {
		log.Fatal("usage: zwfm-speech-segments AUDIO (a file, a URL, or - for standard input)")
	}
	input := os.Args[1]

	began := time.Now()
	step := func(name string) {
		log.Printf("%-12s %5.1f s", name, time.Since(began).Seconds())
		began = time.Now()
	}

	pcm, err := decode(input)
	if err != nil {
		log.Fatal(err)
	}
	if len(pcm) < 16000 {
		log.Fatalf("%q: less than a second of audio", input) //nolint:gosec // G706: %q escapes the value
	}
	step("decode")
	dir, err := model.Dir()
	if err != nil {
		log.Fatal(err)
	}
	frames, feats := features.Compute(pcm)
	step("features")
	xm, err := xvector.Load(dir)
	if err != nil {
		log.Fatal(err)
	}
	wins := xvector.Windows(frames)
	xvs := xm.Extract(frames, feats, wins)
	step("x-vectors")
	dm, err := diarize.Load(dir)
	if err != nil {
		log.Fatal(err)
	}
	segs, err := dm.Run(wins, xvs)
	if err != nil {
		log.Fatal(err)
	}
	step("diarization")

	// Duration is the recording length as the recipe computes it, from the
	// number of 10 ms frames; it bounds the segments.
	out := output{
		Duration: timecode(float64(frames) / 100),
		Segments: numberSpeakers(segs),
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		log.Fatal(err)
	}
}

// decode returns the audio as 16 kHz mono 16-bit PCM, the conversion the
// recipe's README prescribes (ffmpeg -ac 1 -ar 16000). input "-" is read
// from standard input.
func decode(input string) ([]int16, error) {
	// -rw_timeout (microseconds) turns a stalled connection into an error
	// instead of a hang; -xerror makes a broken download fail instead of
	// silently yielding a shorter recording.
	//nolint:gosec // G204: no shell; input is the operator's own argument, passed to ffmpeg as one argument.
	cmd := exec.Command("ffmpeg", "-nostdin", "-v", "error", "-xerror", "-rw_timeout", "30000000", "-i", input,
		"-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-")
	if input == "-" {
		cmd.Stdin = os.Stdin
	}
	cmd.Stderr = os.Stderr
	raw, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffmpeg %q: %w", input, err)
	}
	pcm := make([]int16, len(raw)/2)
	if _, err := binary.Decode(raw, binary.LittleEndian, pcm); err != nil {
		return nil, err
	}
	return pcm, nil
}

// numberSpeakers replaces the segmenter's cluster labels by 1, 2, ... in
// order of total speaking time.
func numberSpeakers(segs []diarize.Segment) []segment {
	total := map[int]float64{}
	for _, s := range segs {
		total[s.Speaker] += s.End - s.Start
	}
	labels := slices.Collect(maps.Keys(total))
	slices.SortFunc(labels, func(a, b int) int {
		return cmp.Or(cmp.Compare(total[b], total[a]), cmp.Compare(a, b))
	})
	number := map[int]int{}
	for i, l := range labels {
		number[l] = i + 1
	}
	out := make([]segment, len(segs))
	for i, s := range segs {
		out[i] = segment{timecode(s.Start), timecode(s.End), number[s.Speaker]}
	}
	return out
}

// timecode formats seconds as HH:MM:SS.mmm.
func timecode(seconds float64) string {
	ms := int64(math.Round(seconds * 1000))
	return fmt.Sprintf("%02d:%02d:%02d.%03d", ms/3600000, ms/60000%60, ms/1000%60, ms%1000)
}
