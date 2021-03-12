#!/usr/bin/env python

""" Distribution configuration """

import imp
import sys

from setuptools import setup, find_packages


VERSION = imp.load_source("brainbuilder.version", "brainbuilder/version.py").VERSION

BASE_REQUIRES = [
    'click>=7.0,<8.0',
    'h5py>=3.1.0',
    'lxml>=3.3',
    'numpy>=1.9',
    'pandas>=1.0.0',
    'scipy>=0.13',
    'tqdm>=4.0',
] + [
    'bluepy>=2.1',
    'libsonata>=0.1.6',
    'voxcell>=3.0.0',
]

SUBCELLULAR_REQUIRES = [
    'attrs<20',   # to use entity-management<1.0, need to use old attrs
    'entity-management>=0.1.11,<1.0',
    'subcellular-querier>=0.0.3',
    'tables>=3.4',
]


REINDEX_REQUIRES = [
    'morphio>=2.3,<3',
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
        'all': SUBCELLULAR_REQUIRES + REINDEX_REQUIRES,
        'subcellular': SUBCELLULAR_REQUIRES,
        'reindex': REINDEX_REQUIRES,
    },
    packages=find_packages(),
    python_requires='>=3.6',
    entry_points={
      'console_scripts': [
          'brainbuilder=brainbuilder.app.__main__:main'
      ]
    }
)
