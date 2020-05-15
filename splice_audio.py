"""
this is just to automate
resources:

[](https://stackoverflow.com/a/23747395/7771202)

Example usage:
>> rm -r example_data/output_silences/cleanest_setup  & python splice_audio.py --silence-split -i example_data/cleanest_setup.weba
or
>> rm -r example_data/output_subtitles/cleanest_setup & python splice_audio.py -i example_data/cleanest_setup.weba --subtitles example_data/cleanest_setup.srt

"""

from argparse import ArgumentParser, FileType
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

import tqdm



def splice_using_silences(args):
    sound_file = AudioSegment.from_file(args.input)
    audio_chunks = split_on_silence(sound_file, 
        # must be silent for at least half a second
        min_silence_len=args.min_silence_len,
        # consider it silent if quieter than -16 dBFS
        silence_thresh=args.silence_thresh,
        keep_silence=1000,
        seek_step=1,
    )

    filename = '.'.join(os.path.split(args.input)[-1].split('.')[:-1])
    os.makedirs(os.path.join(args.out, filename), exist_ok=True)

    for i, chunk in enumerate(audio_chunks):
        out_file = os.path.join(os.path.join(args.out, filename), f"{filename}_{i}.{args.out_fmt}")
        print("exporting", out_file)
        chunk.export(out_file, format=args.out_fmt)


def splice_using_subtitles(args):
    from subs_audio_splicer import Splicer, Parser
    from slugify import slugify
    parser, splicer = Parser(args.subtitles), Splicer(args.input)

    dialogues = parser.get_dialogues()
    audio = AudioSegment.from_file(splicer.audio)

    filename = '.'.join(os.path.split(args.input)[-1].split('.')[:-1])
    os.makedirs(os.path.join(args.out, filename), exist_ok=True)

    with open(os.path.join(os.path.join(args.out, filename), f'{args.speaker_id}-{args.clip_id}.trans.txt') , 'w') as ftrans:
        for i, dialogue in enumerate(tqdm.tqdm(dialogues)):
            start, end = dialogue.start, dialogue.end
            # duration = end-start
            chunk = audio[start: end + 200]

            # TODO: join short audio

            # out_file = os.path.join(os.path.join(args.out, filename), f"_{i}_'{slugify('_'.join(dialogue.text.split(' ')))}'.{args.out_fmt}")
            # chunk.export(out_file, format=args.out_fmt)
            ftrans.write(f'{args.speaker_id}-{args.clip_id}-{i:04d} {dialogue.text.upper()}\n')

            # formatted outfile
            fmt_out_file = os.path.join(os.path.join(args.out, filename), f"{args.speaker_id}-{args.clip_id}-{i:04d}.{args.out_fmt}")
            chunk.export(fmt_out_file, format=args.out_fmt)


if __name__ == "__main__":
    parser = ArgumentParser(description="Given a single sound file, tries to split it to words")
    parser.add_argument('-i', '--input', metavar='AUDIO_INPUT',
                    help='audio file input path')
    parser.add_argument('-o', '--out', metavar='OUT', default='$INPUT$/output_$ACTION$',
                    help='output directory path. Default=$INPUT$/output_$ACTION$ meaning the input directory.'
                    '$ACTION$ means "silences", or "subtitles"')

    # the action: either split based on .rst file, or split based on audio only
    actions_group = parser.add_mutually_exclusive_group(required=True)
    # read .rst file
    actions_group.add_argument('-s', '--silence-split', action='store_true',
                    help='split words based on silences. Can specify --min-silence-len and --silence-thresh')
    actions_group.add_argument('-b', '--subtitles', metavar='SUBTITLES',
                    help='split audio based on subtitle times from the passed .rst/.ass file')

    split_words_group = actions_group.add_argument_group(title='split words', 
                    description='split audio based on words based on silences, must specify --min-silence-len and --silence-thresh')
    split_words_group.add_argument('-msl', '--min-silence-len', default=500, 
                    help='must be silent for at least MIN_SILENCE_LEN (ms)')
    split_words_group.add_argument('-st', '--silence-thresh', default=-16, 
                    help='consider it silent if quieter than SILENCE_THRESH dBFS')

    parser.add_argument('-f', '--out-fmt', metavar='FILE_EXT', default='flac',
                    help='output file extension {mp3, wav, flac, ...}')
    parser.add_argument('--speaker-id', metavar='NUMBER', default=0,
                    help='ID of the speaker clip')
    parser.add_argument('--clip-id', metavar='NUMBER', default=0,
                    help='ID of the source clip')

    args = parser.parse_args()
    args.out = args.out.replace('$INPUT$', os.path.join(*os.path.split(args.input)[:-1]))
    args.out = args.out.replace('$ACTION$', 'silences' if args.silence_split else 'subtitles')
    # args.out = os.path.abspath(args.out)
    # args.input = os.path.abspath(args.input)
    print('args.out', args.out)
    
    os.makedirs(args.out, exist_ok=True)

    if args.silence_split:
        print('using splicing words using silences')
        splice_using_silences(args)
    else:
        print('splicing words using subtitles')
        splice_using_subtitles(args)
