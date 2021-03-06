#!/usr/bin/env python
from sys import platform

import setuptools
from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ['tqdm', 'pydub', 'ass', 'pysrt', 'argcomplete']


if platform == "linux" or platform == "linux2":
    # linux
    pass
elif platform == "darwin":
    # OS X
    pass
elif platform == "win32":
    # Windows...
    requirements += ['gooey']


setuptools.setup(
    name="splice_audio",  # Replace with your own username
    version="1.0",
    author="Faris Hijazi",
    author_email="theefaris@gmail.com",
    description='Audio preprocessing script. Splits audio files to segments using subtitle files or on silences.'
                '\nSpecifically for transcribed audio files.'
                '\nThis is a preprocessing step for speech datasets (specifically LibriSpeech).'
                '\nAnd will generate a ".trans.txt" file.'
                '\nGooey GUI is used if it is installed and no arguments are passed.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FarisHijazi/splice_audio",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=['splice_audio/splice_audio', 'splice_audio/splice_audio.py'],
    python_requires='>=3',
    install_requires=requirements,
    # extras_require={"": ['gooey', 'argcomplete']},
)
