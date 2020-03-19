#!/usr/bin/env python

""" Distribution configuration """

import imp
import sys

from setuptools import setup, find_packages


if sys.version_info < (2, 7):
    sys.exit("Python < 2.7 is no longer supported from version 0.1.0")

VERSION = imp.load_source("brainbuilder.version", "brainbuilder/version.py").VERSION

BASE_REQUIRES = [
    'click>=7.0,<8.0',
    'future>=0.16',
    'h5py>=2.6',
    'lxml>=3.3',
    'numpy>=1.9',
    'pandas>=0.17',
    'pyyaml',
    'scipy>=0.13',
    'six>=1.0',
    'tqdm>=4.0',
] + [
    'bluepy>=0.13.0',
    'voxcell[sonata]>=2.6.2,<3.0',
    'morph-tool>=0.2.3',
]

SUBCELLULAR_REQUIRES = [
    'entity-management>=0.1.11,<1.0',
    'subcellular-querier>=0.0.3',
    'tables>=3.4',
    'pathlib2>=2.3.3'
]


setup(
    name='brainbuilder',
    author='NSE Team',
    author_email='bbp-ou-nse@groupes.epfl.ch',
    version=VERSION,
    description='BrainBuilder is a tool to define the cells that will compose a circuit.',
    url='https://bbpteam.epfl.ch/project/issues/projects/BRBLD/issues/',
    download_url='https://bbpteam.epfl.ch/repository/devpi/+search?query=name%3Abrainbuilder',
    license='BBP-internal-confidential',
    install_requires=BASE_REQUIRES,
    extras_require={
        'all': SUBCELLULAR_REQUIRES,
        'subcellular': SUBCELLULAR_REQUIRES,
    },
    packages=find_packages(),
    entry_points={
      'console_scripts': [
          'brainbuilder=brainbuilder.app.__main__:main'
      ]
    }
)
