# Test data

The golden tests compare each stage of this port with the original Kaldi recipe on `audio.wav`.

`audio.wav` is a clip of 2 minutes 25 seconds, 16 kHz mono. Three LibriVox readers take turns, with two stretches of music and two seconds of near-silence in between.

The speech comes from [Verzamelde korte Nederlandstalige Werken 002](https://librivox.org/verzamelde-korte-nederlandstalige-werken-002-by-various/), a LibriVox recording that is in the public domain ([CC0](https://creativecommons.org/publicdomain/zero/1.0/), see the [archive.org item](https://archive.org/details/verzameld_nederlands_002_1102_librivox)):

- "De door prof. Roentgen ontdekte stralen" by H.A. Lorentz, read by Anna Simon
- "Een Huwelijk" by Arij Prins, read by Marcel Coenders
- "Een Misdaad" by Marcellus Emants, read by Bart de Leeuw

The music is synthetic: sine-wave chords that `make.sh` generates with ffmpeg. A faint white-noise floor, also generated, runs under the whole clip. Without it Kaldi's random dither, which this port cannot reproduce, would decide the quiet mel bands.

The other files are the output of [bbc-speech-segmenter](https://github.com/bbc/bbc-speech-segmenter) on the clip. `mfcc`, `cmvn`, `xvectors` and `plda_scores` are Kaldi text exports, gzipped. `seg/` holds the recipe's own files that the tests read.

To rebuild everything, run `./make.sh`. It needs curl, ffmpeg and Docker. The committed files were made with ffmpeg 9.0.2. Another ffmpeg version may decode or mix the audio slightly differently, and the reference output then changes with it.
