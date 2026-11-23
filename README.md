# zwfm-speech-segments

Finds where people speak in an hour of radio, and groups the speech by voice. It reads an hour from the ZuidWest audio logger and prints the speech segments as JSON.

It is a Go port of [bbc-speech-segmenter](https://github.com/bbc/bbc-speech-segmenter) by BBC Research & Development. It runs the same model through the same steps, without Kaldi, Python or Docker. On a four-core x86 server an hour takes about ten seconds, plus the download.

## Install

You need Go 1.27 or newer and ffmpeg.

```sh
go install github.com/oszuidwest/zwfm-speech-segments@latest
```

For a server, build one static binary and copy it over:

```sh
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build   # or GOARCH=arm64
```

The first run downloads the model, about 35 MB, from the bbc-speech-segmenter repository. It is pinned to one commit, checked against SHA-256 hashes and kept in your user cache directory.

## Usage

```sh
zwfm-speech-segments 2026-11-23_08            # ZuidWest FM, 08:00 to 09:00
zwfm-speech-segments 2026-11-23_08 rucphen    # Radio Rucphen
zwfm-speech-segments 2026-11-23_08 > 08.json
```

The hour is in Dutch local time, as the audio logger names its files. Progress goes to stderr, the JSON to stdout:

```json
{
  "station": "zuidwest",
  "start": "2026-11-23T08:00:00+01:00",
  "duration": "00:59:59.990",
  "segments": [
    { "start": "00:00:00.000", "end": "00:00:12.188", "speaker": 4 },
    { "start": "00:01:18.750", "end": "00:01:20.250", "speaker": 4 }
  ]
}
```

Segment times count from the start of the hour. Between segments the model heard no speech: music, jingles or silence. Speaker 1 is the voice with the most speaking time in that hour. The model groups similar voices; it does not know who is speaking, and the numbers only hold within one hour.

## How it works

1. ffmpeg decodes the hour to 16 kHz mono.
2. The audio becomes MFCC features, with Kaldi's settings and rounding.
3. A neural network turns every 1.5 seconds of audio, in steps of 0.75 seconds, into an x-vector: a fingerprint of the voice.
4. A classifier keeps the x-vectors that contain speech. All x-vectors are scored against each other with PLDA and clustered into voices. Neighbouring windows are then merged into segments.

The heavy matrix work runs on hand-written AVX2 kernels on x86 and NEON kernels on ARM, in Go assembly. No cgo or external library is involved.

## Accuracy

Every step was checked against a run of the original Kaldi recipe on the same hour. Given the same input, the x-vectors match to a cosine similarity of 0.999999997, and the final segments match exactly.

From MP3 to JSON, speech and non-speech agree with the original at least 99.9 percent of the time. Speaker numbers agree less, between roughly 80 and 99 percent of the speech time. Kaldi adds a little random noise to the audio (dither) that cannot be reproduced, and the clustering reacts to small differences like that: runs of this port with different noise differ from each other just as much. Treat the speaker numbers as a rough grouping.

## Development

```sh
go test ./...
```

Some tests compare each stage with reference output of the original Kaldi recipe. They read it from `data/zuidwest/2026-11-23_08` and are skipped when it is not there.

## Credits

This project stands on the work of others:

- [bbc-speech-segmenter](https://github.com/bbc/bbc-speech-segmenter) by Matt Haynes and Misa Ogura at BBC Research & Development: the recipe, the speech classifier and the packaged model. The method is described in M. Ogura and M. Haynes, "X-vector-vad for Multi-genre Broadcast Speech-to-text" (2021).
- [Kaldi](https://github.com/kaldi-asr/kaldi), whose feature extraction, x-vector extraction, PLDA scoring and clustering this code translates.
- The x-vector and PLDA model come from the DIHARD 2019 baseline and are distributed with bbc-speech-segmenter.

## License

Apache License 2.0, the same license as Kaldi and bbc-speech-segmenter. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
