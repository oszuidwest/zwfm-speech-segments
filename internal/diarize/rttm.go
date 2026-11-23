package diarize

import "strconv"

// turn is one RTTM line: start, duration and the cluster label. The values
// are the ones the recipe's scripts hold after parsing the previous file.
type turn struct {
	Beg, Dur float64
	Label    int
}

// round3 is a "%.3f" print followed by float(): what each RTTM hop does.
func round3(x float64) float64 {
	v, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 3, 64), 64)
	return v
}

// makeRTTM is local/make_rttm.py on the VAD-filtered segments: overlapping
// windows are cut at the midpoint, then touching windows with the same
// label are joined.
func makeRTTM(wins []Window, keep []bool, labels []int) []turn {
	type seg struct {
		start, end float64
		label      int
	}
	var segs []seg
	for i, w := range wins {
		if keep[i] {
			segs = append(segs, seg{w.Start, w.End, labels[i]})
		}
	}
	if len(segs) == 0 {
		return nil
	}
	var cont []seg
	for i := 0; i < len(segs)-1; i++ {
		s, next := segs[i], &segs[i+1]
		if s.end > next.start {
			avg := (next.start + s.end) / 2.0
			next.start = avg
			s.end = avg
		}
		cont = append(cont, s)
	}
	cont = append(cont, segs[len(segs)-1])
	var out []turn
	for i := 0; i < len(cont)-1; i++ {
		s, next := cont[i], &cont[i+1]
		if s.end == next.start && s.label == next.label {
			next.start = s.start
			continue
		}
		out = append(out, turn{round3(s.start), round3(s.end - s.start), s.label})
	}
	s := cont[len(cont)-1]
	return append(out, turn{round3(s.start), round3(s.end - s.start), s.label})
}

// mergeSingle is local/merge_single_xvector_clusters.py: runs of turns of
// at most 0.75 s join the longer neighbour.
func mergeSingle(turns []turn) []turn {
	queue := append([]turn(nil), turns...)
	var out []turn
	for len(queue) > 0 {
		c := queue[0]
		queue = queue[1:]
		if c.Dur > 0.75 {
			out = append(out, c)
			continue
		}
		for len(queue) > 0 && queue[0].Dur <= 0.75 {
			c.Dur += queue[0].Dur
			queue = queue[1:]
		}
		switch {
		case len(out) == 0 && len(queue) == 0:
			out = append(out, c) // the script raises IndexError here
		case len(out) == 0:
			queue[0].Beg, queue[0].Dur = c.Beg, queue[0].Dur+c.Dur
		case len(queue) == 0 || out[len(out)-1].Dur >= queue[0].Dur:
			out[len(out)-1].Dur += c.Dur
		default:
			queue[0].Beg, queue[0].Dur = c.Beg, queue[0].Dur+c.Dur
		}
	}
	return out
}

// padSegments is local/pad_segments.py: 0.75 s of padding on each side
// (none after the last turn, none before 0), overlaps cut at the midpoint,
// touching turns of one speaker joined. The result is rounded like its
// "%0.3f" output.
func padSegments(turns []turn) []turn {
	const pad = 0.75
	u := append([]turn(nil), turns...)
	for i := range u {
		if i == 0 {
			u[i].Beg = max(0, u[i].Beg-pad)
		} else {
			u[i].Beg -= pad
		}
		if i < len(u)-1 {
			u[i].Dur += pad
		}
	}
	for i := 0; i < len(u)-1; i++ {
		if u[i].Beg+u[i].Dur > u[i+1].Beg {
			avg := (u[i].Beg + u[i].Dur + u[i+1].Beg) / 2
			u[i].Dur = avg - u[i].Beg
			u[i+1].Beg = avg
		}
	}
	var out []turn
	for i := range u {
		if i < len(u)-1 && u[i+1].Label == u[i].Label && u[i].Beg+u[i].Dur == u[i+1].Beg {
			u[i+1].Beg = u[i].Beg
			u[i+1].Dur += u[i].Dur
			continue
		}
		out = append(out, turn{round3(u[i].Beg), round3(u[i].Dur), u[i].Label})
	}
	return out
}
