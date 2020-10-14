#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
this is just to automate splicing (splitting audio on silences or)
resources: [](https://stackoverflow.com/a/23747395/7771202)


Example usage:
```
$ python splice_audio.py -f -i data/my_audio.wav 
# or pass a subtitles file
$ python splice_audio.py -f -i data/my_audio.wav --subtitles data/my_subtitles.srt


more extensive example: (assumes you have $audioname and $name defined)

$ python splice_audio.py -i "data/$audioname.wav" --out="{INPUT}/LibriSpeech/train-$name/0" --subtitles="data/$audioname.srt" --subtitle_end_offset=200 --subtitle_rescale=1
```
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from pathlib import Path

from pydub import AudioSegment
from pydub.utils import db_to_float
from tqdm import tqdm


# === pydub function ===

def detect_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1, disable_tqdm=True):
    seg_len = len(audio_segment)

    # you can't have a silent portion of a sound that is longer than the sound
    if seg_len < min_silence_len:
        return []

    # convert silence threshold to a float value (so we can compare it to rms)
    silence_thresh = db_to_float(silence_thresh) * audio_segment.max_possible_amplitude

    # find silence and add start and end indicies to the to_cut list
    silence_starts = []

    # check successive (1 sec by default) chunk of sound for silence
    # try a chunk at every "seek step" (or every chunk for a seek step == 1)
    last_slice_start = seg_len - min_silence_len
    slice_starts = range(0, last_slice_start + 1, seek_step)

    # guarantee last_slice_start is included in the range
    # to make sure the last portion of the audio is searched
    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])

    for i in tqdm(slice_starts, desc='slicing silences', disable=disable_tqdm):
        audio_slice = audio_segment[i:i + min_silence_len]
        if audio_slice.rms <= silence_thresh:
            silence_starts.append(i)

    # short circuit when there is no silence
    if not silence_starts:
        return []

    # combine the silence we detected into ranges (start ms - end ms)
    silent_ranges = []

    prev_i = silence_starts.pop(0)
    current_range_start = prev_i

    for silence_start_i in silence_starts:
        continuous = (silence_start_i == prev_i + seek_step)

        # sometimes two small blips are enough for one particular slice to be
        # non-silent, despite the silence all running together. Just combine
        # the two overlapping silent ranges.
        silence_has_gap = silence_start_i > (prev_i + min_silence_len)

        if not continuous and silence_has_gap:
            silent_ranges.append([current_range_start,
                                  prev_i + min_silence_len])
            current_range_start = silence_start_i
        prev_i = silence_start_i

    silent_ranges.append([current_range_start,
                          prev_i + min_silence_len])

    return silent_ranges


def detect_nonsilent(audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1, disable_tqdm=True):
    silent_ranges = detect_silence(audio_segment, min_silence_len, silence_thresh, seek_step, disable_tqdm=disable_tqdm)
    len_seg = len(audio_segment)
    # if there is no silence, the whole thing is nonsilent
    if not silent_ranges:
        return [[0, len_seg]]

    # short circuit when the whole audio segment is silent
    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
        return []

    prev_end_i = 0
    nonsilent_ranges = []
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append([prev_end_i, start_i])
        prev_end_i = end_i

    if end_i != len_seg:
        nonsilent_ranges.append([prev_end_i, len_seg])

    if nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)

    return nonsilent_ranges


def split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,
                     seek_step=1, disable_tqdm=True, output_ranges=None):
    """
    audio_segment - original pydub.AudioSegment() object

    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms

    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS

    keep_silence - (in ms or True/False) leave some silence at the beginning
        and end of the chunks. Keeps the sound from sounding like it
        is abruptly cut off.
        When the length of the silence is less than the keep_silence duration
        it is split evenly between the preceding and following non-silent
        segments.
        If True is specified, all the silence is kept, if False none is kept.
        default: 100ms
    """

    # from the itertools documentation
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    output_ranges = output_ranges or [
        [start - keep_silence, end + keep_silence]
        for (start, end)
        in detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step, disable_tqdm=disable_tqdm)
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end + next_start) // 2
            range_ii[0] = range_i[1]

    return [
               audio_segment[max(start, 0): min(end, len(audio_segment))]
               for start, end in output_ranges
           ], output_ranges


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0  # ms
    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


# === silence functions ===

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


# === helpers ===

def _check_samplerate(args):
    # check samplerate
    out = subprocess.Popen(['ffmpeg', '-i', args.i.as_posix()],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    stdout = stdout.decode('utf8')
    stderr = stderr.decode('utf8') if stderr is not None else None

    lines = stdout.split('\n')
    segments = [line for line in lines if " Hz, " in line][0].split(', ')
    number = [seg for seg in segments if " Hz" in seg][0].split(' ')[0]

    input_samplerate = int(number)
    if input_samplerate > args.samplerate and \
            (args.force or input(
                f'Samplerate {input_samplerate} not matching, create {args.samplerate}Hz version? [(Y)es/no]').lower() in [
                 'n', 'no']):
        input_resampled = args.i.as_posix() + f'.{args.samplerate}Hz' + args.i.suffix
        out = subprocess.Popen(['ffmpeg', '-i', args.i.as_posix(), '-ar', f'{args.samplerate}', input_resampled],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
        stdout = stdout.decode('utf8')
        stderr = stderr.decode('utf8') if stderr is not None else None

        print(stdout)
        if stderr or 'unable' in stdout:
            print(f"Couldn't convert {args.i}")
            raise Exception(stderr)
        else:
            args.i = Path(input_resampled)
            assert args.i.exists(), f'{args.i} does not exist! Failed to create {args.samplerate}Hz version'


def _force_short_chunks(audio_chunks, max_len=15000, min_silence_len=500, silence_thresh=-16, scaledown=0.9,
                        pbar=None) -> list:
    # ensure that all chunks are less than the max len:
    min_silence_len, silence_thresh = int(round(min_silence_len * scaledown)), silence_thresh * scaledown
    short_chunks = []
    if pbar is None:
        pbar = tqdm(audio_chunks, position=0, leave=False, total=len(audio_chunks))
    else:
        pbar.total += len(audio_chunks)

    for audio_chunk in audio_chunks:
        pbar.update(1)
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
                    subsplits, output_ranges = split_on_silence(
                        audio_chunk,
                        min_silence_len=min_silence_len,
                        silence_thresh=silence_thresh,
                        keep_silence=1000,
                        seek_step=1,
                    )
                    short_chunks += _force_short_chunks(subsplits, max_len, min_silence_len, silence_thresh, pbar=pbar)
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
            # combined_len = subchunks_len + len(next_chunk)
            i += 1

            # if already too big, discard
            if not (subchunks_len <= max_len):
                # pbar.set_description(f'\rthrow chunk of length {subchunks_len}')
                # subchunks.append(next_chunk)  # NOTE: for debugging, normally wouldn't be here
                break
            else:  # if can merge more
                # pbar.set_description('\rappended chunk {} \t to chunks \t[{}]=\t{}'.format(
                #     len(next_chunk),
                #     '+'.join(list(map(str, map(len, subchunks)))),
                #     combined_len + len(next_chunk),
                # ))
                subchunks.append(next_chunk)
        else:  # can't add more, just merge them and move on
            # next_chunk = audio_chunks[i - 1]
            # subchunks_len = sum(list(map(len, subchunks)))
            # combined_len = subchunks_len + len(next_chunk)

            # pbar.set_description('\rmerged chunks [{}] =\t {} E {}'.format(
            #     '+'.join(list(map(str, map(len, subchunks)))),
            #     combined_len,
            #     (min_len, max_len)
            # ))
            merged_chunks.append(sum(subchunks))
    else:
        pbar.close()
    return merged_chunks


# === core functions ===

def splice_using_silences(input: Path, out: Path, min_silence_len=500, silence_thresh=-16, out_fmt='flac', min_len=7000,
                          max_len=15000, **kwargs):
    filename = out.name.replace(' ', '-')
    out_parent_dir = out.parent.name
    out_dir = out

    print('loading audio...')
    sound_file = AudioSegment.from_file(input)

    # try to load an existing meta,
    output_ranges = None

    try:
        metas = json.load(out_dir.joinpath(f'{out_parent_dir}-{filename}.meta.json').open(mode='r'))
        output_ranges = metas['output_ranges']
        print(f'found existing output_ranges in meta file containing {len(output_ranges)} ranges!')
    except:
        pass

    audio_chunks, output_ranges = split_on_silence(
        sound_file,
        # must be silent for at least half a second
        min_silence_len=min_silence_len,
        # consider it silent if quieter than -16 dBFS
        silence_thresh=silence_thresh,
        keep_silence=1000,
        seek_step=1,
        disable_tqdm=False,
        output_ranges=output_ranges
    )

    metas = dict(
        output_ranges=output_ranges,
        files=[],
    )

    print('\nforce_short_chunks()')
    audio_chunks = _force_short_chunks(audio_chunks, max_len, min_silence_len, silence_thresh)
    lens = sum(list(map(len, audio_chunks)))
    print(
        f'\nafter force_short_chunks: segments  / total_original: {lens}/{len(sound_file)} = {lens / len(sound_file) * 100:.2f}% kept')

    # audio_chunks = [remove_leading_and_trailing_silences(chunk, silence_thresh) for chunk in tqdm(audio_chunks, 'remove_leading_and_trailing_silences')]
    # lens = sum(list(map(len, audio_chunks)))
    # print(f'\nafter remove_leading_and_trailing_silences: segments  / total_original: {lens}/{len(sound_file)} = {lens / len(sound_file) * 100:.2f}% kept')

    # audio_chunks = [_remove_silences(chunk, min_silence_len) for chunk in tqdm(audio_chunks, '_remove_silences')]
    # lens = sum(list(map(len, audio_chunks)))
    # print(f'\nafter _remove_silences(): segments  / total_original: {lens}/{len(sound_file)} = {lens / len(sound_file) * 100:.2f}% kept')

    # TODO: FIXME: this isn't preserving the order, later make an option to be ordered or not
    print('\nmerge_short_segments_to_range()')
    # merged_chunks = _merge_short_segments_to_range(
    #     [chunk for chunk in audio_chunks if (len(chunk) < max_len)], min_len, max_len) + \
    #         [chunk for chunk in audio_chunks if not (len(chunk) < max_len)]
    merged_chunks = _merge_short_segments_to_range(audio_chunks, min_len, max_len)

    lens = sum(list(map(len, merged_chunks)))
    postmerge_time_perc = lens / len(sound_file) * 100
    print(
        f'\nafter merge chunks: all segments  / total_original: {lens}/{len(sound_file)} = {postmerge_time_perc:.2f}% kept')

    valid_chunks = [chunk for chunk in merged_chunks if (min_len > len(chunk) < max_len)]
    # invalid_chunks = [chunk for chunk in merged_chunks if not (min_len > len(chunk) < max_len)]
    invalid_chunks = [chunk for chunk in merged_chunks if not (len(chunk) < max_len)]

    metas['n_files'] = len(merged_chunks),
    metas['postmerge_time_perc'] = postmerge_time_perc,

    with out_dir.joinpath(f'{out_parent_dir}-{filename}.trans.txt').open(mode='w') as ftrans:
        for i, chunk in enumerate(merged_chunks):
            fmt_out_file = out_dir.joinpath(f"{out_parent_dir}-{filename}-{i:04d}.{out_fmt}")

            # if not in range, put in special directory
            if not (min_len <= len(chunk) <= max_len):
                not_in_range = fmt_out_file.parent.joinpath('not_in_range')
                not_in_range.mkdir(exist_ok=True)
                fmt_out_file = not_in_range.joinpath(fmt_out_file.name)

            metas['files'].append({
                'path': fmt_out_file.as_posix(),
                'length': len(chunk),
            })
            chunk.export(fmt_out_file, format=out_fmt)
            text = ''
            ftrans.write(f'{filename}-{i:04d} {text}\n')

    metas['total_seconds'] = sum(f['length'] for f in metas['files'])
    json.dump(metas,
              out_dir.joinpath(f'{out_parent_dir}-{filename}.meta.json').open(mode='w'),
              indent=4)


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
        dialogue.text = re.sub(r'\s+', ' ', re.sub(r"[^a-zA-z0-9': ]+", ' ', dialogue.text)).strip()

    for i, dialogue in enumerate(dialogues):
        next_start = dialogues[i + 1].start if i < len(dialogues) - 1 else 999999999999999
        dialogue.start, dialogue.end = dialogue.start, min(next_start, dialogue.end) + subtitle_end_offset

    filename = out.name.replace(' ', '-')
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
            ftrans.write(f'{filename}-{i:04d} {text}\n')

            # formatted outfile
            fmt_out_file = out_dir.joinpath(f"{filename}-{i:04d}.{out_fmt}".replace(' ', '-'))
            chunk.export(fmt_out_file, format=out_fmt)
    print(f'Segments have covered {(length / len(audio)) * 100:.2f}%')


def go(args):
    print('args', args)
    method = 'sub' if args.subtitles else 'sil'
    input_parent = args.i.parent
    filename = '.'.join(args.i.name.split('.')[:-1])
    args.out_fmt = args.out_fmt or args.i.name.split('.')[-1]

    if args.subtitles and args.subtitles.exists():
        if args.force or input(f'WARNING: passed subtitle file does not exist "{args.subtitles}"'
                               f'split with silences? [yes/(N)o]').lower() in ['y', 'yes']:
            args.subtitles = None
        else:
            print('exiting')
            return 1

    print('variables:', {'{INPUT}': input_parent, '{NAME}': filename, '{METHOD}': method})

    args.o = Path(
        args.o.as_posix().format(INPUT=input_parent,
                                 NAME=filename,
                                 METHOD=method,
                                 )
    ).joinpath(filename)

    try:
        # noinspection PyUnresolvedReferences
        from argutils import print_args

        print_args(args)
    except:
        pass

    # if output already exists, check user or check if forced
    existing_files = list(args.o.glob('*[!.meta.json]'))
    if not args.child and args.o.is_dir() and existing_files:
        print(f'{args.o} already exists and is nonempty, choose the "--force" option to overwrite it.')
        if (args.force or input(f'overwrite "{args.o}"? [yes/(N)o]').lower() in ['y', 'yes']):
            print('DELETING', args.o)
            list(map(os.remove, existing_files))
        else:
            print('exiting')
            return 1
    else:
        args.o.mkdir(parents=True, exist_ok=True)

    args.input = args.i
    args.out = args.o
    # check samplerate
    try:
        if args.samplerate > 0:
            _check_samplerate(args)
        else:
            print("Skipping samplerate check.")
    except Exception as e:
        print('Warning: Could not check sample using ffmpeg', e)
    # take action
    if args.subtitles:
        print(f'splicing "{args.i}" using subtitles')
        splice_using_subtitles(**vars(args))
    else:
        print(f'splicing "{args.i}" using silences. You can use --subtitles')
        splice_using_silences(**vars(args))

    print('saved to output directory:', args.o)
    return 0


def main():
    try:
        import gooey
    except ImportError:
        gooey = None
        print("INFO: Failed to import Gooey (GUI disabled)")

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
                        'Gooey GUI is used if it is installed and no arguments are passed.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('-i', metavar='INPUT_AUDIO', type=Path, widget='FileChooser', nargs='+',
                            gooey_options={
                                'validator': {
                                    'test': 'user_input != None',
                                    'message': 'Choose a valid file path'
                                }
                            },
                            help='audio file input path')
        parser.add_argument('-o', metavar='OUT_FOLDER', type=Path, widget='DirChooser',
                            default='{INPUT}/splits-{METHOD}',
                            gooey_options={
                                'validator': {
                                    'test': 'user_input != None',
                                    'message': 'Choose a valid directory path'
                                }
                            },
                            help='output directory path. '
                                 '{INPUT} means the input directory. '
                                 '{METHOD} means "sil" (silences), or "sub" (subtitles)')
        parser.add_argument('--n_procs', default=cpu_count(), type=int,
                            help='Multiprocessing concurrency level (best to leave at default).')
        parser.add_argument('--samplerate', default=0, type=int,
                            help='Assert target samplerate. If 0 then any samplerate is allowed.')
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
                                          'If the audio segments say more than the text, then make this smaller. ')

        group_subtitles.add_argument('--subtitle_rescale', default=1.0, type=float,
                                     help='rescale the timings if the audio was rescaled (stretched).'
                                          'Example1: If subtitle_rescale=2, then it will finish in half the original time.'
                                          'Example2: If the subtitle dialogues are ahead (appear earlier) of the audio, increase subtitle_rescale.')

        group_silences = parser.add_argument_group('Silences')
        group_silences.add_argument('-sl', '--min_silence_len', default=700, type=int,
                                    help='must be silent for at least MIN_SILENCE_LEN (ms) to count as a silence. ')
        group_silences.add_argument('-st', '--silence_thresh', default=-50, type=float,
                                    help='consider it silent if quieter than SILENCE_THRESH dBFS. ')

        parser.add_argument('--min_len', default=6000, type=int,
                            help='minimum length for each segment in ms. ')
        parser.add_argument('--max_len', default=13000, type=int,
                            help='maximum length for each segment in ms. ')

        parser.add_argument('--out_fmt', metavar='FILE_EXT', default='',
                            help='output file extension {"", mp3, wav, flac, ...}. Empty means same as input')

        parser.add_argument('-f', '--force', action='store_true',
                            help='Overwrite if output already exists')
        return parser

    parser = get_parser()

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except:
        print("INFO: failed to import `argcomplete` lib")

    args = parser.parse_args(sys.argv[1:])

    # expand directory
    input_dir = args.i[0]
    if len(args.i) == 1 and input_dir.is_dir():
        args.i = list(input_dir.rglob('*.mp3'))
        assert len(args.i) > 0, 'No mp3 files found in input directory:'
        print('input is a folder, found mp3 files:', args.i)

    try:
        # noinspection PyUnresolvedReferences
        from argutils import print_args

        print_args(args)
    except:
        pass

    # make args for each input file
    args_list = []
    for i in args.i:
        args_clone = deepcopy(args)
        args_clone.i = i
        args_clone.child = len(args.i) > 1
        args_list.append(args_clone)

    n_procs = args.n_procs
    ret_codes = list(Pool(n_procs).map(go, args_list))

    return max(ret_codes)


if __name__ == "__main__":
    main()
