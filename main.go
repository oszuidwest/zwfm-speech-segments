// Command zwfm-speech-segments prints the speech segments and speakers in one
// hour of ZuidWest FM or Radio Rucphen as JSON.
//
// Usage:
//
//	zwfm-speech-segments YYYY-MM-DD_HH [zuidwest|rucphen]
//
// For example:
//
//	zwfm-speech-segments 2026-11-23_08            ZuidWest FM, 08:00-09:00
//	zwfm-speech-segments 2026-11-23_08 rucphen    Radio Rucphen, 08:00-09:00
//
// The hour is read in Dutch time and downloaded from audio.zuidwest.cloud.
// Each segment has a start and end time code and a speaker number; speaker 1
// is the voice with the most speaking time.
//
// It is a Go port of the BBC speech segmenter
// (github.com/bbc/bbc-speech-segmenter): Kaldi MFCC and sliding CMN, the
// x-vector network, the calibrated VAD classifier, PLDA scoring,
// agglomerative clustering and the recipe's post-processing.
//
// ffmpeg must be installed and on the PATH; it decodes the MP3.
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
	_ "time/tzdata"

	"github.com/oszuidwest/zwfm-speech-segments/internal/diarize"
	"github.com/oszuidwest/zwfm-speech-segments/internal/features"
	"github.com/oszuidwest/zwfm-speech-segments/internal/model"
	"github.com/oszuidwest/zwfm-speech-segments/internal/xvector"
)

// output is the JSON document written to stdout.
type output struct {
	Station  string    `json:"station"`
	Start    time.Time `json:"start"`
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
	if len(os.Args) < 2 || len(os.Args) > 3 {
		log.Fatal("usage: zwfm-speech-segments YYYY-MM-DD_HH [zuidwest|rucphen]")
	}
	hour, station := os.Args[1], "zuidwest"
	if len(os.Args) == 3 {
		station = os.Args[2]
	}
	amsterdam, err := time.LoadLocation("Europe/Amsterdam")
	if err != nil {
		log.Fatal(err)
	}
	start, err := time.ParseInLocation("2006-01-02_15", hour, amsterdam)
	if err != nil {
		log.Fatalf("hour %q: want YYYY-MM-DD_HH", hour) //nolint:gosec // G706: %q escapes the value
	}
	if start.Format("2006-01-02_15") != hour {
		log.Fatalf("hour %q does not exist in Dutch time (clocks go forward)", hour) //nolint:gosec // G706: %q escapes the value
	}

	began := time.Now()
	step := func(name string) {
		log.Printf("%-12s %5.1f s", name, time.Since(began).Seconds())
		began = time.Now()
	}

	url := fmt.Sprintf("https://audio.zuidwest.cloud/%s/%s.mp3", station, hour)
	pcm, err := decode(url)
	if err != nil {
		log.Fatal(err)
	}
	if len(pcm) < 16000 {
		log.Fatalf("%s: less than a second of audio", url) //nolint:gosec // G706: hour is validated above; station is a CLI argument from the operator
	}
	step("download")
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
	segs := dm.Run(wins, xvs)
	step("diarization")

	// Duration is the recording length as the recipe computes it, from the
	// number of 10 ms frames; it bounds the segments.
	out := output{
		Station:  station,
		Start:    start,
		Duration: timecode(float64(frames) / 100),
		Segments: numberSpeakers(segs),
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		log.Fatal(err)
	}
}

// decode returns 16 kHz mono 16-bit PCM, the same conversion the recipe's
// README prescribes (ffmpeg -ac 1 -ar 16000).
func decode(url string) ([]int16, error) {
	// -rw_timeout (microseconds) turns a stalled connection into an error
	// instead of a hang; -xerror makes a broken download fail instead of
	// silently yielding a shorter hour.
	//nolint:gosec // G204: no shell; url is a fixed https host plus the validated hour and the operator's station argument.
	cmd := exec.Command("ffmpeg", "-nostdin", "-v", "error", "-xerror", "-rw_timeout", "30000000", "-i", url,
		"-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-")
	cmd.Stderr = os.Stderr
	raw, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffmpeg %s: %w", url, err)
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
