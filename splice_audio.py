"""
this is just to automate
resources:

[](https://stackoverflow.com/a/23747395/7771202)

Example usage:
```
$ python splice_audio.py -f -i data/my_audio.wav 
# or pass a subtitles file
$ python splice_audio.py -f -i data/my_audio.wav --subtitles data/my_subtitles.srt


more extensive example: (assumes you have $audioname and $name defined)

$ python splice_audio.py -i "data/$audioname.wav" --out="\$INPUT\$/LibriSpeech/train-$name/0" --subtitles="data/$audioname.srt" --subtitle_end_offset=200 --subtitle_rescale=1
```
"""

import os
from argparse import ArgumentParser

from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_leading_silence, detect_nonsilent
from pathlib import Path

from utils.argutils import print_args


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
                    pbar.set_description(f'force_short_chunks({len(audio_chunk)/1000:.2f}sec)')
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
                except UnboundLocalError as e:
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
            elif combined_len <= max_len:  # if can merge more
                pbar.set_description('\rappended chunk {} \t to chunks \t[{}]=\t{}'.format(
                    len(next_chunk),
                    '+'.join(list(map(str, map(len, subchunks)))),
                    combined_len + len(next_chunk),
                ))
                subchunks.append(next_chunk)
            else:
                break
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


def splice_using_silences(input: Path, out: Path, silence_thresh=500, min_silence_len=-50,
                          out_fmt='flac', min_len=7000, max_len=15000, **kwargs):
    filename = '.'.join(input.name.split('.')[:-1])
    out_dir = out.joinpath(filename)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parent_dir = out.name

    sound_file = AudioSegment.from_file(input)
    audio_chunks = split_on_silence(
        sound_file,
        # must be silent for at least half a second
        min_silence_len=min_silence_len,
        # consider it silent if quieter than -16 dBFS
        silence_thresh=silence_thresh,
        keep_silence=1000,
        seek_step=1,
    )

    print('force_short_chunks()')
    audio_chunks = _force_short_chunks(audio_chunks, max_len, min_silence_len, silence_thresh)

    print('merge_short_segments_to_range()')
    merged_chunks = _merge_short_segments_to_range(audio_chunks, min_len, max_len)

    original_len = len(sound_file)
    lens = list(map(len, merged_chunks))
    print(f'Total_trimmed / total_original: {sum(lens)}/{original_len} = {sum(lens) / original_len * 100:.2f}%')

    for i, chunk in enumerate(merged_chunks):
        fmt_out_file = out_dir.joinpath(f"{out_parent_dir}-{filename}-{i:04d}.{out_fmt}")
        chunk.export(fmt_out_file, format=out_fmt)


def splice_using_subtitles(input: Path, out: Path, subtitles: Path, silence_thresh=500, min_silence_len=-50,
                           out_fmt='flac', subtitle_rescale=1.0, subtitle_end_offset=400, **kwargs):
    from subs_audio_splicer import Splicer, Parser
    parser_, splicer = Parser(subtitles.as_posix()), Splicer(input.as_posix())
    dialogues = parser_.get_dialogues()

    # in the case of a speedup factor, change the subtitle times
    for dialogue in dialogues:
        dialogue.start = dialogue.start / subtitle_rescale
        dialogue.end = dialogue.end / subtitle_rescale

    filename = '.'.join(input.name.split('.')[:-1])
    out_dir = out.joinpath(filename)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parent_dir = out.name

    print('filename:', out_dir.joinpath(f'{out_parent_dir}-{filename}.trans.txt'))
    length = 0

    audio = AudioSegment.from_file(splicer.audio)
    with open(out_dir.joinpath(f'{out_parent_dir}-{filename}.trans.txt').as_posix(), 'w') as ftrans:
        pbar = tqdm(dialogues)
        for i, dialogue in enumerate(pbar):
            next_start = dialogues[i + 1].start if i < len(dialogues) - 1 else 999999999999999
            # next_end = dialogues[i + 1].end if i < len(dialogues) - 1 else 0
            # nextnext_end = dialogues[i + 2].end if i < len(dialogues) - 2 else 0
            if not dialogue.text:
                continue

            start, end = dialogue.start, min(next_start, dialogue.end) + subtitle_end_offset
            # duration = end-start
            chunk = audio[start: end]

            # removing leading and trailing silences
            start_trim = detect_leading_silence(chunk, silence_threshold=silence_thresh)
            end_trim = detect_leading_silence(chunk.reverse(), silence_threshold=silence_thresh)
            chunk = chunk[start_trim:len(chunk) - end_trim]
            pbar.set_description(f"striping silences in range ({start_trim}, {end_trim})")
            # remove silences
            chunk = _remove_silences(chunk, min_silence_len, silence_thresh)
            length += len(chunk)

            ftrans.write(f'{out_parent_dir}-{filename}-{i:04d} {dialogue.text.upper()}\n')

            # formatted outfile
            fmt_out_file = out_dir.joinpath(f"{out_parent_dir}-{filename}-{i:04d}.{out_fmt}")
            chunk.export(fmt_out_file, format=out_fmt)
    print(f'Segments have covered {(length / len(audio)) * 100:.2f}%')


def _remove_silences(sound: AudioSegment, min_silence_len=50, silence_thresh=None):
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
    parser = ArgumentParser(
        description='Given a single sound file, tries to split it to segments.'
                    'This is a preprocessing step for speech datasets (specifically LibriSpeech).'
                    'It can split on silences or split using a subtitles file. and will generate a ".trans.txt" file')

    parser.add_argument('-i', '--input', metavar='INPUT_AUDIO', type=Path,
                        help='audio file input path')
    parser.add_argument('-o', '--out', metavar='OUT_FOLDER', type=Path,
                        default='$INPUT$/LibriSpeech/train-clean-100-$ACTION$/0',
                        help='output directory path. '
                             'Default="$INPUT$/LibriSpeech/train-clean-100-$ACTION$/0". '
                             '$INPUT$ means the input directory. '
                             '$ACTION$ means "sil" (silences), or "sub" (subtitles)')

    # the action: either split based on .rst file, or split based on audio only
    group_subtitles = parser.add_argument_group('Subtitles')
    # read .rst file
    group_subtitles.add_argument('-b', '--subtitles', metavar='SUBTITLES_FILE', type=Path,
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
    group_silences.add_argument('--min_silence_len', default=500, type=int,
                                help='must be silent for at least MIN_SILENCE_LEN (ms) to count as a silence. '
                                     'Default=500')
    group_silences.add_argument('--silence_thresh', default=-50, type=float,
                                help='consider it silent if quieter than SILENCE_THRESH dBFS. '
                                     'Default=-50')

    parser.add_argument('--min_len', default=7000, type=int,
                        help='minimum length for each segment in ms. '
                             'Default=7000.')
    parser.add_argument('--max_len', default=15000, type=int,
                        help='maximum length for each segment in ms. '
                             'Default=15000.')

    parser.add_argument('--out_fmt', metavar='FILE_EXT', default='flac',
                        help='output file extension {mp3, wav, flac, ...}')

    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite if output already exists')
    # parser.add_argument("--hparams", type=str, default="", help=\
    #     "Hyperparameter overrides as a comma-separated list of name-value pairs")

    args = parser.parse_args()
    args.out = Path(
        args.out.as_posix()
            .replace('$INPUT$', os.path.join(*os.path.split(args.input)[:-1]))
            .replace('$ACTION$', 'sub' if args.subtitles else 'sil')
    )

    print_args(args)
    # if output already exists, check user or check if forced
    if args.out.is_dir():
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
