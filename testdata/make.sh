#!/bin/sh
# Rebuilds audio.wav and the Kaldi reference output next to this script.
# Needs curl, ffmpeg and Docker. Kaldi runs as linux/amd64, under emulation
# on ARM.
set -eu
export LC_ALL=C
cd "$(dirname "$0")"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# Three readers from the CC0 LibriVox recording "Verzamelde korte
# Nederlandstalige Werken 002".
src=https://archive.org/download/verzameld_nederlands_002_1102_librivox/nederlandsecollectie002
curl -fsSL -o "$work/a.mp3" "${src}_01_rontgenlorentz_as_64kb.mp3"
curl -fsSL -o "$work/b.mp3" "${src}_02_eenhuwelijk_mc_64kb.mp3"
curl -fsSL -o "$work/c.mp3" "${src}_06_misdaad_bdl_64kb.mp3"

# chord prints a decaying three-note chord with a bass note an octave below.
chord() {
	echo "(sin(2*PI*$1*t)+sin(2*PI*$2*t)+sin(2*PI*$3*t)+sin(PI*$1*t))"
}
# Two synthetic music pieces: C Am F G, 1.25 s per chord, and Am F C G,
# 1 s per chord.
m1="0.08*exp(-1.2*mod(t,1.25))*(lt(mod(t,5),1.25)*$(chord 261.63 329.63 392.00)+between(mod(t,5),1.25,2.4999)*$(chord 220.00 261.63 329.63)+between(mod(t,5),2.5,3.7499)*$(chord 174.61 220.00 261.63)+gte(mod(t,5),3.75)*$(chord 196.00 246.94 293.66))"
m2="0.08*exp(-2*mod(t,1))*(lt(mod(t,4),1)*$(chord 220.00 261.63 329.63)+between(mod(t,4),1,1.9999)*$(chord 174.61 220.00 261.63)+between(mod(t,4),2,2.9999)*$(chord 261.63 329.63 392.00)+gte(mod(t,4),3)*$(chord 196.00 246.94 293.66))"

# Turns A B, music, A C, a short gap, B, music, C. Cuts fall in pauses. A
# white-noise floor of about -42 dBFS runs under everything. Without it many
# mel bands of the music and the readings hold little more than Kaldi's
# dither, which cannot be reproduced, and the features follow the dither.
ffmpeg -nostdin -loglevel error -i "$work/a.mp3" -i "$work/b.mp3" -i "$work/c.mp3" -filter_complex "
[0]aresample=16000,pan=mono|c0=c0,asplit[a1][a2];
[1]aresample=16000,pan=mono|c0=c0,asplit[b1][b2];
[2]aresample=16000,pan=mono|c0=c0,asplit[c1][c2];
[a1]atrim=40.5:58.9,asetpts=N/SR/TB[s1];
[b1]atrim=30.3:51.4,asetpts=N/SR/TB[s2];
aevalsrc='$m1':s=16000:d=10[s3];
[a2]atrim=114.9:137.9,asetpts=N/SR/TB[s4];
[c1]atrim=38.8:56.9,asetpts=N/SR/TB[s5];
anullsrc=r=16000:cl=mono:d=2[s6];
[b2]atrim=92.7:113.9,asetpts=N/SR/TB[s7];
aevalsrc='$m2':s=16000:d=8[s8];
[c2]atrim=92.5:115.5,asetpts=N/SR/TB[s9];
[s1][s2][s3][s4][s5][s6][s7][s8][s9]concat=n=9:v=0:a=1[speech];
anoisesrc=r=16000:a=0.012:seed=1[floor];
[speech][floor]amix=inputs=2:duration=first:normalize=0" \
	-ac 1 -ar 16000 -c:a pcm_s16le -map_metadata -1 -fflags +bitexact "$work/audio.wav"

# Run the recipe on one segment that spans the whole file.
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$work/audio.wav")
printf 'audio 0 audio 0.00 %.2f <label> _\n' "$dur" >"$work/audio.stm"
docker run --rm --platform linux/amd64 -v "$work:/data" \
	bbcrd/bbc-speech-segmenter@sha256:0ecad3b284a0acec24f04d7e45cf14f768cece57d4909454375de77460405ed8 \
	sh -c './run-segmentation.sh /data/audio.wav /data/audio.stm /data/seg &&
		cd /data && PATH=/opt/kaldi/src/featbin:/opt/kaldi/src/bin:/opt/kaldi/src/ivectorbin:$PATH &&
		copy-feats scp:seg/feats.scp ark,t:mfcc.txt &&
		copy-feats scp:seg/cmvn/feats.scp ark,t:cmvn.txt &&
		copy-vector ark:seg/xvectors/xvector.all.ark ark,t:xvectors.txt &&
		copy-matrix scp:seg/plda/scores.scp ark,t:plda_scores.txt'

# Keep only what the tests read, replacing the old files only now that
# every step has worked.
cp "$work/audio.wav" audio.wav
for f in mfcc cmvn xvectors plda_scores; do
	gzip -9n <"$work/$f.txt" >"$f.txt.gz"
done
for f in xvectors/subsegments xvectors/predictions.txt xvectors/segments.filt clusters/labels diarize.stm; do
	mkdir -p "seg/$(dirname "$f")"
	cp "$work/seg/$f" "seg/$f"
done
