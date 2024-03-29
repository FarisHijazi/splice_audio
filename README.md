# SpliceAudio

Audio splicer splices on subtitles or on silences

## Installation

### pip install

simply:

```shell script
pip install git+https://github.com/FarisHijazi/splice_audio
```

### Development installation (optional)

This repo does use another repository: https://github.com/sverrod/subs-audio-splicer

```shell script
git clone https://github.com/FarisHijazi/splice_audio
```

#### Optional installs

Installing autocomplete (optional)

```shell script
pip install argcomplete
activate-global-python-argcomplete --user
```

trying to install wxpython for linux (but these commands don't work yet, feel free to suggest ones that do)

```shell script
sudo add-apt-repository ppa:swt-techie/wxpython4
sudo apt-get update
sudo apt-get install python3-wxgtk4.0
```

## Usage

```
usage: splice_audio    [-h] -i INPUT_AUDIO [INPUT_AUDIO ...] [-o OUT_FOLDER]
                       [--n_procs N_PROCS] [--samplerate SAMPLERATE]
                       [-b SUBTITLES_FILE]
                       [--subtitle_end_offset SUBTITLE_END_OFFSET]
                       [--subtitle_rescale SUBTITLE_RESCALE]
                       [-sl MIN_SILENCE_LEN] [-st SILENCE_THRESH]
                       [--min_len MIN_LEN] [--max_len MAX_LEN]
                       [--out_fmt FILE_EXT] [-f]

Given a single sound file, tries to split it to segments.This is a
preprocessing step for speech datasets (specifically LibriSpeech).It can split
on silences or using a subtitles file. And will generate a ".trans.txt"
file.Gooey GUI is used if it is installed and no arguments are passed.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_AUDIO [INPUT_AUDIO ...]
                        audio file input path (default: None)
  -o OUT_FOLDER         output directory path. {INPUT} means the input
                        directory. {METHOD} means "sil" (silences), or "sub"
                        (subtitles) (default: {INPUT}/splits-{METHOD})
  --n_procs N_PROCS     Multiprocessing concurrency level (best to leave at
                        default). (default: 32)
  --samplerate SAMPLERATE
                        Assert target samplerate. If 0 then any samplerate is
                        allowed. (default: 0)
  --min_len MIN_LEN     minimum length for each segment in ms. (default: 6000)
  --max_len MAX_LEN     maximum length for each segment in ms. (default:
                        13000)
  --out_fmt FILE_EXT    output file extension {mp3, wav, flac, ...} (default:
                        flac)
  -f, --force           Overwrite if output already exists (default: False)

Subtitles:
  -b SUBTITLES_FILE, --subtitles SUBTITLES_FILE
                        split audio based on subtitle times from the passed
                        .rst/.ass fileIf not passed, will split words based on
                        silences. Can specify --min_silence_len and
                        --silence_thresh (default: None)
  --subtitle_end_offset SUBTITLE_END_OFFSET
                        add delay at end of subtitle dialogue (milliseconds).
                        If the audio segments say more than the text, then
                        make this smaller. (default: 100)
  --subtitle_rescale SUBTITLE_RESCALE
                        rescale the timings if the audio was rescaled
                        (stretched).Example1: If subtitle_rescale=2, then it
                        will finish in half the original time.Example2: If the
                        subtitle dialogues are ahead (appear earlier) of the
                        audio, increase subtitle_rescale. (default: 1.0)

Silences:
  -sl MIN_SILENCE_LEN, --min_silence_len MIN_SILENCE_LEN
                        must be silent for at least MIN_SILENCE_LEN (ms) to
                        count as a silence. (default: 700)
  -st SILENCE_THRESH, --silence_thresh SILENCE_THRESH
                        consider it silent if quieter than SILENCE_THRESH
                        dBFS. (default: -50)

```

Uses Gooey graphical interface (if installed).

![](audio_splicer_gui.png)

## Known issues

- gitsubmodule error `ERROR: Command errored out with exit status 128: git submodule update --init --recursive -q Check the logs for full command output.` Still working on the fix.
- issues with Windows install including Gooey. Solution: pip install it manually

