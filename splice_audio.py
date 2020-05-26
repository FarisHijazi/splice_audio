"""
this is just to automate
resources:

[](https://stackoverflow.com/a/23747395/7771202)

Example usage:
```
$ python splice_audio.py -f --silence_split -i example_data/cleanest_setup.wav
# or
$ python splice_audio.py -f -i example_data/cleanest_setup.wav --subtitles example_data/cleanest_setup.srt

```
"""

from argparse import ArgumentParser, FileType
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_leading_silence, detect_nonsilent

import os
import tqdm

from utils.argutils import print_args


def splice_using_silences(input, out, silence_thresh=500, min_silence_len=-50,
                          out_fmt='flac', **kwargs):

    filename = '.'.join(os.path.split(input)[-1].split('.')[:-1])
    out_dir = os.path.join(out, filename)
    os.makedirs(out_dir, exist_ok=True)
    out_parent_dir = os.path.split(out)[-1]
    
    sound_file = AudioSegment.from_file(input)
    audio_chunks = split_on_silence(sound_file, 
        # must be silent for at least half a second
        min_silence_len=min_silence_len,
        # consider it silent if quieter than -16 dBFS
        silence_thresh=silence_thresh,
        keep_silence=1000,
        seek_step=1,
    )

    filename = '.'.join(os.path.split(args.input)[-1].split('.')[:-1])
    out_parent_dir = os.path.join(args.out, filename)
    os.makedirs(out_parent_dir, exist_ok=True)

    for i, chunk in enumerate(audio_chunks):
        fmt_out_file = os.path.join(out_dir, f"{out_parent_dir}-{filename}-{i:04d}.{out_fmt}")
        chunk.export(fmt_out_file, format=out_fmt)


def splice_using_subtitles(input, out, subtitles, silence_thresh=500, min_silence_len=-50,
                           out_fmt='flac', subtitle_rescale=1.0, **kwargs):
    from subs_audio_splicer import Splicer, Parser
    from slugify import slugify
    parser, splicer = Parser(subtitles), Splicer(input)

    dialogues = parser.get_dialogues()

    # in the case of a speedup factor, change the subtitle times
    for dialogue in dialogues:
        dialogue.start = dialogue.start / subtitle_rescale
        dialogue.end = dialogue.end / subtitle_rescale

    filename = '.'.join(os.path.split(input)[-1].split('.')[:-1])
    out_dir = os.path.join(out, filename)
    os.makedirs(out_dir, exist_ok=True)
    out_parent_dir = os.path.split(out)[-1]

    print('filename:', os.path.join(out_dir, f'{out_parent_dir}-{filename}.trans.txt'))

    audio = AudioSegment.from_file(splicer.audio)
    with open(os.path.join(out_dir, f'{out_parent_dir}-{filename}.trans.txt') , 'w') as ftrans:
        for i, dialogue in enumerate(tqdm.tqdm(dialogues)):
            next_start = dialogues[i+1].start if i < len(dialogues)-1 else 999999999999999
            start, end = dialogue.start, min(next_start, dialogue.end)
            # duration = end-start
            chunk = audio[start: end]

            # removing leading and trailing silences
            start_trim = detect_leading_silence(chunk, silence_threshold=silence_thresh)
            end_trim = detect_leading_silence(chunk.reverse(), silence_threshold=silence_thresh)
            chunk = chunk[start_trim:len(chunk)-end_trim]
            # remove silences
            # chunk = remove_silences(chunk, min_silence_len, silence_thresh)

            ftrans.write(f'{out_parent_dir}-{filename}-{i:04d} {dialogue.text.upper()}\n')

            # formatted outfile
            fmt_out_file = os.path.join(out_dir, f"{out_parent_dir}-{filename}-{i:04d}.{out_fmt}")
            chunk.export(fmt_out_file, format=out_fmt)

            fmt_out_file = os.path.join(out_dir, f"{out_parent_dir}-{filename}-{i:04d}.{args.out_fmt}")
            chunk.export(fmt_out_file, format=args.out_fmt)



if __name__ == "__main__":
    parser = ArgumentParser(description="Given a single sound file, tries to split it to words")
    parser.add_argument('-i', '--input', metavar='AUDIO_INPUT',
                    help='audio file input path')
    parser.add_argument('-o', '--out', metavar='OUT', default='$INPUT$/LibriSpeech/train-clean-100-$ACTION$/0',
                    help='output directory path.'\
                    'Default=$INPUT$/LibriSpeech/train-clean-100-$ACTION$/0 meaning the input directory.'
                    '$ACTION$ means "sil" (silences), or "sub" (subtitles)')

    # the action: either split based on .rst file, or split based on audio only
    actions_group = parser.add_mutually_exclusive_group(required=True)
    # read .rst file
    actions_group.add_argument('-s', '--silence_split', action='store_true',
                    help='split words based on silences.'\
                        'Can specify --min_silence_len and --silence_thresh')
    actions_group.add_argument('-b', '--subtitles', metavar='SUBTITLES',
                    help='split audio based on subtitle times from the passed .rst/.ass file')

    parser.add_argument('--subtitle_rescale', default=1.0, type=float,
                    help='rescale the timings if the audio was rescaled. default=1'\
                        'If subtitle_rescale=2, then it will finish in half the original time.')
    parser.add_argument('-msl', '--min_silence_len', default=500, type=float,
                    help='must be silent for at least MIN_SILENCE_LEN (ms)')
    parser.add_argument('-st', '--silence_thresh', default=-50,  type=float,
                    help='consider it silent if quieter than SILENCE_THRESH dBFS')
    parser.add_argument('--out_fmt', metavar='FILE_EXT', default='flac',
                    help='output file extension {mp3, wav, flac, ...}')
    
    parser.add_argument('-f', '--force', action='store_true',
                    help='Overwrite if output already exists')

    args = parser.parse_args()
    args.out = args.out.replace('$INPUT$', os.path.join(*os.path.split(args.input)[:-1]))
    args.out = args.out.replace('$ACTION$', 'sil' if args.silence_split else 'sub')

    # if output already exists, check user or check if forced
    if os.path.isdir(args.out):
        if args.force or input(f'overwrite "{args.out}"? [yes/(N)o]').lower() in ['y', 'yes']:
            print('deleting', args.out)
            import shutil
            shutil.rmtree(args.out)
        else:
            print('exiting')
            exit(0)
    else:
        os.makedirs(args.out)

    os.makedirs(args.out, exist_ok=True)

    if args.silence_split:
        print('using splicing words using silences')
        splice_using_silences(**vars(args))
    else:
        print('splicing words using subtitles')
        splice_using_subtitles(**vars(args))

    print('saved to output directory:', args.out)
