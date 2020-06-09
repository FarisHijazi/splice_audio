"""
this is just to automate
resources:

[](https://stackoverflow.com/a/23747395/7771202)

Example usage:
```
$ python splice_audio.py -f --silence_split -i data/my_audio.wav
# or
$ python splice_audio.py -f -i data/my_audio.wav --subtitles data/my_audio.srt

```
"""

import os
from argparse import ArgumentParser

from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_leading_silence, detect_nonsilent

from utils.argutils import print_args


def force_short_chunks(audio_chunks, max_len=15000, min_silence_len=100, silence_thresh=-50.0, scaledown=0.9) -> list:
    # ensure that all chunks are less than the max len:
    min_silence_len, silence_thresh = int(round(min_silence_len * scaledown)), silence_thresh * scaledown
    short_chunks = []
    for audio_chunk in tqdm(audio_chunks):
        if len(audio_chunk) < max_len:
            short_chunks.append(audio_chunk)
        else:
            # if too long, split again
            local_min_silence_len, local_silence_thresh = min_silence_len, silence_thresh
            # keep trying to split, 
            for _ in range(100):
                try:
                    # this thing might raise an error
                    subsplits = split_on_silence(
                        audio_chunk,
                        min_silence_len=min_silence_len,
                        silence_thresh=silence_thresh,
                        keep_silence=1000,
                        seek_step=1,
                    )
                    short_chunks += force_short_chunks(subsplits, max_len, min_silence_len, silence_thresh)
                    break
                except UnboundLocalError as e:
                    local_min_silence_len = int(round(local_min_silence_len * scaledown))
                    local_silence_thresh  = local_silence_thresh * scaledown
                    print(f'scaling down min_silence_len={local_min_silence_len}, silence_thresh={local_silence_thresh}', end='\r')
            else:
                short_chunks += list(filter(lambda x: len(x)<max_len, subsplits))

    return short_chunks


def splice_using_silences(input, out, silence_thresh=500, min_silence_len=-50,
                          out_fmt='flac', min_len=7000, max_len=15000, **kwargs):
    filename = '.'.join(os.path.split(input)[-1].split('.')[:-1])
    out_dir = os.path.join(out, filename)
    os.makedirs(out_dir, exist_ok=True)
    out_parent_dir = os.path.split(out)[-1]

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
    audio_chunks = force_short_chunks(audio_chunks, max_len, min_silence_len, silence_thresh)

    print('merge_short_segments_to_range()')
    merged_chunks = merge_short_segments_to_range(audio_chunks, min_len, max_len)


    original_len = len(sound_file)
    lens = list(map(len, merged_chunks))
    print(f'Total_trimmed / total_original: {sum(lens)}/{original_len} = {sum(lens) / original_len * 100:.2f}%')

    for i, chunk in enumerate(merged_chunks):
        fmt_out_file = os.path.join(out_dir, f"{out_parent_dir}-{filename}-{i:04d}.{out_fmt}")
        chunk.export(fmt_out_file, format=out_fmt)

def merge_short_segments_to_range(audio_chunks, min_len, max_len):
    """
    given short audio chunks, will merge them until they are in the range (min_len, max_len)
    NOTE: assumes all chunks are shorter than the max_len

    @param audio_chunks: list of pydub AudioSegments
    @param min_len: min length for a segment
    @param max_len: max length for a segment
    @return:
    """
    merged_chunks = []
    pbar = tqdm(desc='merging splits', total=len(audio_chunks))
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
                # print(f'throw chunk of length {subchunks_len}')
                break
            elif combined_len <= max_len:  # if can merge more
                # print('appended subchunk {} \t to subchunks \t[{}]=\t{}'.format(
                #     len(next_chunk),
                #     '+'.join(list(map(str,map(len, subchunks)))),
                #     combined_len + len(next_chunk),
                # ))
                subchunks.append(next_chunk)
            else:
                break
        else:  # can't add more, just merge them and move on
            # print('merged chunks [{}] =\t {} E {}'.format(
            #     '+'.join(list(map(str,map(len, subchunks)))),
            #     combined_len,
            #     (min_len, max_len)
            # ))
            merged_chunks.append(sum(subchunks))
    else:
        pbar.close()
    lens = list(map(len, merged_chunks))
    print(f'Lengths of merged_chunks: {lens}')
    return merged_chunks


def splice_using_subtitles(input, out, subtitles, silence_thresh=500, min_silence_len=-50,
                           out_fmt='flac', subtitle_rescale=1.0, **kwargs):
    from subs_audio_splicer import Splicer, Parser
    parser_, splicer = Parser(subtitles), Splicer(input)

    dialogues = parser_.get_dialogues()

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
    with open(os.path.join(out_dir, f'{out_parent_dir}-{filename}.trans.txt'), 'w') as ftrans:
        for i, dialogue in enumerate(tqdm(dialogues)):
            next_start = dialogues[i + 1].start if i < len(dialogues) - 1 else 999999999999999
            # next_end = dialogues[i + 1].end if i < len(dialogues) - 1 else 0
            # nextnext_end = dialogues[i + 2].end if i < len(dialogues) - 2 else 0

            start, end = dialogue.start, min(next_start, dialogue.end) + 400
            # duration = end-start
            chunk = audio[start: end]

            # removing leading and trailing silences
            start_trim = detect_leading_silence(chunk, silence_threshold=silence_thresh)
            end_trim = detect_leading_silence(chunk.reverse(), silence_threshold=silence_thresh)
            chunk = chunk[start_trim:len(chunk) - end_trim]
            # print("striping silences: (start_trim, end_trim)=", (start_trim, end_trim))
            # remove silences
            # chunk = remove_silences(chunk, min_silence_len, silence_thresh)

            ftrans.write(f'{out_parent_dir}-{filename}-{i:04d} {dialogue.text.upper()}\n')

            # formatted outfile
            fmt_out_file = os.path.join(out_dir, f"{out_parent_dir}-{filename}-{i:04d}.{out_fmt}")
            chunk.export(fmt_out_file, format=out_fmt)



if __name__ == "__main__":
    parser = ArgumentParser(description="Given a single sound file, tries to split it to words")
    parser.add_argument('-i', '--input', metavar='AUDIO_INPUT',
                        help='audio file input path')
    parser.add_argument('-o', '--out', metavar='OUT', default='$INPUT$/LibriSpeech/train-clean-100-$ACTION$/0',
                        help='output directory path.' \
                             'Default=$INPUT$/LibriSpeech/train-clean-100-$ACTION$/0 meaning the input directory.'
                             '$ACTION$ means "sil" (silences), or "sub" (subtitles)')

    # the action: either split based on .rst file, or split based on audio only
    actions_group = parser.add_mutually_exclusive_group(required=True)
    # read .rst file
    actions_group.add_argument('-s', '--silence_split', action='store_true',
                               help='split words based on silences.' \
                                    'Can specify --min_silence_len and --silence_thresh')
    actions_group.add_argument('-b', '--subtitles', metavar='SUBTITLES',
                               help='split audio based on subtitle times from the passed .rst/.ass file')

    parser.add_argument('--subtitle_rescale', default=1.0, type=float,
                        help='rescale the timings if the audio was rescaled. default=1' \
                             'If subtitle_rescale=2, then it will finish in half the original time.')
    parser.add_argument('-msl', '--min_silence_len', default=500, type=int,
                        help='must be silent for at least MIN_SILENCE_LEN (ms) to count as a silence. Default=500')
    parser.add_argument('-st', '--silence_thresh', default=-50, type=float,
                        help='consider it silent if quieter than SILENCE_THRESH dBFS. Default=-50')
    parser.add_argument('--out_fmt', metavar='FILE_EXT', default='flac',
                        help='output file extension {mp3, wav, flac, ...}')

    parser.add_argument('--min_len', default=7000, type=int,
                        help='minimum length for each segment in ms. Default=7000.' \
                             'Only used with split on silences.')
    parser.add_argument('--max_len', default=15000, type=int,
                        help='maximum length for each segment in ms. Default=15000.' \
                             'Only used with split on silences.')

    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite if output already exists')
    # parser.add_argument("--hparams", type=str, default="", help=\
    #     "Hyperparameter overrides as a comma-separated list of name-value pairs")

    args = parser.parse_args()
    args.out = args.out.replace('$INPUT$', os.path.join(*os.path.split(args.input)[:-1]))
    args.out = args.out.replace('$ACTION$', 'sil' if args.silence_split else 'sub')
    print_args(args)
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

    # from synthesizer.hparams import hparams
    # args.hparams = hparams.parse(args.hparams)

    if args.silence_split:
        print('splicing using silences')
        splice_using_silences(**vars(args))
    else:
        print('splicing using subtitles')
        splice_using_subtitles(**vars(args))

    print('saved to output directory:', args.out)
