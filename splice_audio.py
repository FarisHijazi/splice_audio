"""
this is just to automate splicing (splitting audio on silences or)
resources: [](https://stackoverflow.com/a/23747395/7771202)


Example usage:
```
$ python splice_audio.py -f -i data/my_audio.wav 
# or pass a subtitles file
$ python splice_audio.py -f -i data/my_audio.wav --subtitles data/my_subtitles.srt


more extensive example: (assumes you have $audioname and $name defined)

$ python splice_audio.py -i "data/$audioname.wav" --out="<INPUT>/LibriSpeech/train-$name/0" --subtitles="data/$audioname.srt" --subtitle_end_offset=200 --subtitle_rescale=1
```
"""

import argparse
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.utils import db_to_float
from tqdm import tqdm



def _force_short_chunks(audio_chunks, max_len=15000, min_silence_len=100, silence_thresh=-50.0, scaledown=0.9) -> list:
    # ensure that all chunks are less than the max len:
    min_silence_len, silence_thresh = int(round(min_silence_len * scaledown)), silence_thresh * scaledown
    short_chunks = []
    pbar = tqdm(audio_chunks, position=0, leave=False)
    for audio_chunk in pbar:
        if len(audio_chunk) < max_len:
            short_chunks.append(audio_chunk)
        else:
            # if too long, split again
            local_min_silence_len, local_silence_thresh = min_silence_len, silence_thresh
            # keep trying to split, 100 times
            for _ in range(100):
                try:
                    pbar.set_description(f'force_short_chunks({len(audio_chunk) / 1000:.2f}sec)')
                    # this thing might raise an error
                    subsplits = split_on_silence(
                        audio_chunk,
                        min_silence_len=min_silence_len,
                        silence_thresh=silence_thresh,
                        keep_silence=1000,
                        seek_step=1,
                    )
                    short_chunks += _force_short_chunks(subsplits, max_len, min_silence_len, silence_thresh)
                    break
                except UnboundLocalError:
                    local_min_silence_len = int(round(local_min_silence_len * scaledown))
                    local_silence_thresh = local_silence_thresh * scaledown
                    pbar.set_description(f'scaling down min_silence_len={local_min_silence_len:.5f}, '
                                         f'silence_thresh={local_silence_thresh:.5f} ...')
            else:
                short_chunks += list(filter(lambda x: len(x) < max_len, subsplits))

    return short_chunks


def _merge_short_segments_to_range(audio_chunks, min_len, max_len):
    """
    given short audio chunks, will merge them until they are in the range (min_len, max_len)
    NOTE: assumes all chunks are shorter than the max_len

    @param audio_chunks: list of pydub AudioSegments
    @param min_len: min length for a segment
    @param max_len: max length for a segment
    @return:
    """
    print('merging splits')
    merged_chunks = []

    pbar = tqdm(desc='merging splits', total=len(audio_chunks), position=0, leave=True)
    # merge short chunks
    i = 0
    while i < len(audio_chunks):
        subchunks = []  # subchunks in the same list will become one chunk
        # cum chunks until in range
        while i < len(audio_chunks) and not (min_len < sum(list(map(len, subchunks))) < max_len):
            pbar.update(1)
            next_chunk = audio_chunks[i]

            # if connecting will be too big
            subchunks_len = sum(list(map(len, subchunks)))
            combined_len = subchunks_len + len(next_chunk)
            i += 1

            # if already too big, discard
            if not (subchunks_len <= max_len):
                pbar.set_description(f'\rthrow chunk of length {subchunks_len}')
                break
            else:  # if can merge more
                pbar.set_description('\rappended chunk {} \t to chunks \t[{}]=\t{}'.format(
                    len(next_chunk),
                    '+'.join(list(map(str, map(len, subchunks)))),
                    combined_len + len(next_chunk),
                ))
                subchunks.append(next_chunk)
        else:  # can't add more, just merge them and move on
            next_chunk = audio_chunks[i-1]
            subchunks_len = sum(list(map(len, subchunks)))
            combined_len = subchunks_len + len(next_chunk)

            pbar.set_description('\rmerged chunks [{}] =\t {} E {}'.format(
                '+'.join(list(map(str, map(len, subchunks)))),
                combined_len,
                (min_len, max_len)
            ))
            merged_chunks.append(sum(subchunks))
    else:
        pbar.close()
    lens = list(map(len, merged_chunks))
    print(f'Lengths of merged_chunks: {lens}')
    return merged_chunks


def splice_using_silences(input: Path, out: Path, min_silence_len=500, silence_thresh=-16, out_fmt='flac', min_len=7000,
                          max_len=15000, **kwargs):
    filename = out.name
    out_parent_dir = out.parent.name
    out_dir = out

    print('loading audio...')
    sound_file = AudioSegment.from_file(input)
    print('spitting...')
    audio_chunks = split_on_silence(
        sound_file,
        # must be silent for at least half a second
        min_silence_len=min_silence_len,
        # consider it silent if quieter than -16 dBFS
        silence_thresh=silence_thresh,
        keep_silence=1000,
        seek_step=1,
        disable_tqdm=False,
    )

    print('force_short_chunks()')
    audio_chunks = _force_short_chunks(audio_chunks, max_len, min_silence_len, silence_thresh)

    print('merge_short_segments_to_range()')
    merged_chunks = _merge_short_segments_to_range(audio_chunks, min_len, max_len)

    original_len = len(sound_file)
    lens = list(map(len, merged_chunks))
    print(f'Total_trimmed / total_original: {sum(lens)}/{original_len} = {sum(lens) / original_len * 100:.2f}%')

    with out_dir.joinpath(f'{out_parent_dir}-{filename}.trans.txt').open(mode='w') as ftrans:
        for i, chunk in enumerate(merged_chunks):
            fmt_out_file = out_dir.joinpath(f"{out_parent_dir}-{filename}-{i:04d}.{out_fmt}")
            if not (min_len >= len(chunk) <= max_len):
                not_in_range = fmt_out_file.parent.joinpath('not_in_range')
                not_in_range.mkdir(exist_ok=True)
                fmt_out_file = not_in_range.joinpath(fmt_out_file.name)

            chunk.export(fmt_out_file, format=out_fmt)
            text = ''
            ftrans.write(f'{out_parent_dir}-{filename}-{i:04d} {text}\n')


def splice_using_subtitles(input: Path, out: Path, subtitles: Path, min_silence_len=500, silence_thresh=-16,
                           out_fmt='flac', subtitle_rescale=1.0, subtitle_end_offset=400, min_len=7000,
                           max_len=15000, **kwargs):
    from subs_audio_splicer import Splicer, Parser
    parser_, splicer = Parser(subtitles.as_posix()), Splicer(input.as_posix())
    dialogues = [dialogue for dialogue in parser_.get_dialogues() if dialogue.text]

    # in the case of a speedup factor, change the subtitle times
    for dialogue in dialogues:
        dialogue.start = int(round(dialogue.start / subtitle_rescale))
        dialogue.end = int(round(dialogue.end / subtitle_rescale))
        dialogue.text = re.sub(r'\s+', ' ', re.sub(r"[^a-zA-z']", ' ', dialogue.text)).strip()

    for i, dialogue in enumerate(dialogues):
        next_start = dialogues[i + 1].start if i < len(dialogues) - 1 else 999999999999999
        dialogue.start, dialogue.end = dialogue.start, min(next_start, dialogue.end) + subtitle_end_offset

    filename = out.name
    out_parent_dir = out.parent.name
    out_dir = out

    print('filename:', out_dir.joinpath(f'{out_parent_dir}-{filename}.trans.txt'))
    length = 0

    audio = AudioSegment.from_file(splicer.audio)

    dialogues = _merge_short_segments_to_range(dialogues, min_len, max_len)
    lens = sum(list(map(len, dialogues)))
    print(
        f'\nafter merge chunks: all dialogues  / total_original: {lens}/{len(audio)} = {lens / len(audio) * 100:.2f}% kept')

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_dir.joinpath(f'{out_parent_dir}-{filename}.trans.txt').open(mode='w') as ftrans:
        pbar = tqdm(dialogues)
        for i, dialogue in enumerate(pbar):
            next_start = dialogues[i + 1].start if i < len(dialogues) - 1 else 999999999999999
            # next_end = dialogues[i + 1].end if i < len(dialogues) - 1 else 0
            # nextnext_end = dialogues[i + 2].end if i < len(dialogues) - 2 else 0
            # start, end = dialogue.start, min(next_start, dialogue.end) + subtitle_end_offset
            start, end = dialogue.start, dialogue.end

            # duration = end-start
            chunk = audio[start: end]

            # remove silences for each chunk
            chunk = remove_leading_and_trailing_silences(chunk, silence_thresh, keep_silence=50, pbar=pbar)
            chunk = _remove_silences(chunk, min_silence_len, silence_thresh)
            length += len(chunk)

            text = dialogue.text.replace('\n', ' ').upper()
            ftrans.write(f'{out_parent_dir}-{filename}-{i:04d} {text}\n')

            # formatted outfile
            fmt_out_file = out_dir.joinpath(f"{out_parent_dir}-{filename}-{i:04d}.{out_fmt}")
            chunk.export(fmt_out_file, format=out_fmt)
    print(f'Segments have covered {(length / len(audio)) * 100:.2f}%')


def remove_leading_and_trailing_silences(chunk, silence_thresh=-16, keep_silence=50, pbar=None):
    # removing leading and trailing silences
    start_trim = detect_leading_silence(chunk, silence_threshold=silence_thresh)
    end_trim = len(chunk) - detect_leading_silence(chunk.reverse(), silence_threshold=silence_thresh)

    chunk = chunk[max(0, start_trim - keep_silence): min(len(chunk), end_trim + keep_silence)]
    if pbar is not None:
        pbar.set_description(f"striping silences in range ({start_trim}, {end_trim})")
    # remove silences
    return chunk


def _remove_silences(sound: AudioSegment, min_silence_len=500, silence_thresh=None):
    # https://stackoverflow.com/a/60212565
    non_sil_times = detect_nonsilent(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh or sound.dBFS * 1.5
    )
    if non_sil_times:
        sound = sum(sound[start:end] for start, end in non_sil_times)

    return sound




if __name__ == "__main__":
    try:
        import gooey
    except ImportError:
        gooey = None


    def flex_add_argument(f):
        """Make the add_argument accept (and ignore) the widget option."""

        def f_decorated(*args, **kwargs):
            kwargs.pop('widget', None)
            kwargs.pop('gooey_options', None)
            return f(*args, **kwargs)

        return f_decorated


    # Monkey-patching a private class…
    argparse._ActionsContainer.add_argument = flex_add_argument(argparse.ArgumentParser.add_argument)

    # Do not run GUI if it is not available or if command-line arguments are given.
    if gooey is None or len(sys.argv) > 1:
        ArgumentParser = argparse.ArgumentParser


        def gui_decorator(f):
            return f
    else:
        print('Using Gooey')
        ArgumentParser = gooey.GooeyParser
        gui_decorator = gooey.Gooey(
            program_name='Audio splicer',
            progress_regex=r"(\d+)%",
            navigation='TABBED',
            suppress_gooey_flag=True,
        )
        # in gooey mode, --force is always set
        sys.argv.append('--force')


    @gui_decorator
    def get_parser():
        parser = ArgumentParser(
            description='Given a single sound file, tries to split it to segments.'
                        'This is a preprocessing step for speech datasets (specifically LibriSpeech).'
                        'It can split on silences or using a subtitles file. And will generate a ".trans.txt" file.'
                        'Note that it is advised to have 16000Hz audio files as input.'
                        'Gooey GUI is used if it is installed and no arguments are passed.')

        parser.add_argument('-i', '--input', metavar='INPUT_AUDIO', type=Path, widget='FileChooser', required=True,
                            gooey_options={
                                'validator': {
                                    'test': 'user_input != None',
                                    'message': 'Choose a valid file path'
                                }
                            },
                            help='audio file input path')
        parser.add_argument('-o', '--out', metavar='OUT_FOLDER', type=Path, widget='DirChooser',
                            default='<INPUT>/splits-<ACTION>/',
                            gooey_options={
                                'validator': {
                                    'test': 'user_input != None',
                                    'message': 'Choose a valid directory path'
                                }
                            },
                            help='output directory path. '
                                 'Default="<INPUT>/splits-<ACTION>/". '
                                 '<INPUT> means the input directory. '
                                 '<ACTION> means "sil" (silences), or "sub" (subtitles)')

        # the action: either split based on .rst file, or split based on audio only
        group_subtitles = parser.add_argument_group('Subtitles')
        # read .rst file
        group_subtitles.add_argument('-b', '--subtitles', metavar='SUBTITLES_FILE', type=Path,
                                     widget='FileChooser',
                                     gooey_options={'validator': {
                                         'test': 'user_input != None',
                                         'message': 'Choose a valid path'
                                     }},
                                     help='split audio based on subtitle times from the passed .rst/.ass file'
                                          'If not passed, will split words based on silences. '
                                          'Can specify --min_silence_len and --silence_thresh')
        group_subtitles.add_argument('--subtitle_end_offset', default=100, type=int,
                                     help='add delay at end of subtitle dialogue (milliseconds). '
                                          'If the audio segments say more than the text, then make this smaller. '
                                          'Default=100')

        group_subtitles.add_argument('--subtitle_rescale', default=1.0, type=float,
                                     help='rescale the timings if the audio was rescaled.'
                                          'If subtitle_rescale=2, then it will finish in half the original time. '
                                          'Default=1')

        group_silences = parser.add_argument_group('Silences')
        group_silences.add_argument('-sl', '--min_silence_len', default=700, type=int,
                                    help='must be silent for at least MIN_SILENCE_LEN (ms) to count as a silence. '
                                         'Default=700')
        group_silences.add_argument('-st', '--silence_thresh', default=-50, type=float,
                                    help='consider it silent if quieter than SILENCE_THRESH dBFS. '
                                         'Default=-50')

        parser.add_argument('--min_len', default=7000, type=int,
                            help='minimum length for each segment in ms. '
                                 'Default=7000.')
        parser.add_argument('--max_len', default=13000, type=int,
                            help='maximum length for each segment in ms. '
                                 'Default=13000.')

        parser.add_argument('--out_fmt', metavar='FILE_EXT', default='flac',
                            help='output file extension {mp3, wav, flac, ...}')

        parser.add_argument('-f', '--force', action='store_true',
                            help='Overwrite if output already exists')
        return parser


    parser = get_parser()

    args = parser.parse_args(sys.argv[1:])
    args.out = Path(
        args.out.as_posix()
            .replace('<INPUT>', os.path.join(*os.path.split(args.input)[:-1]))
            .replace('<ACTION>', 'sub' if args.subtitles else 'sil')
    ).joinpath(
        # filename
        '.'.join(args.input.name.split('.')[:-1])
    )

    try:
        from utils.argutils import print_args

        print_args(args)
    except:
        pass

    # if output already exists, check user or check if forced
    if args.out.is_dir():
        print(f'{args.out} already exists, choose the "force" option to overwrite it.')
        if args.force or input(f'overwrite "{args.out}"? [yes/(N)o]').lower() in ['y', 'yes']:
            print('DELETING', args.out)
            import shutil

            shutil.rmtree(args.out)
        else:
            print('exiting')
            exit(0)
    else:
        args.out.mkdir(parents=True, exist_ok=True)

    # from synthesizer.hparams import hparams
    # args.hparams = hparams.parse(args.hparams)

    if args.subtitles and args.subtitles.exists():
        print('splicing using subtitles')
        splice_using_subtitles(**vars(args))
    else:
        print('splicing using silences. You can use --subtitles')
        splice_using_silences(**vars(args))

    print('saved to output directory:', args.out)
