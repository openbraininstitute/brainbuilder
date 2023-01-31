#!/usr/bin/env python

import importlib.util

from setuptools import setup, find_packages

spec = importlib.util.spec_from_file_location(
    "brainbuilder.version",
    "brainbuilder/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.__version__

BASE_REQUIRES = [
    'click>=7.0,<9.0',
    'h5py>=3.1.0',
    'jsonschema>=3.2.0',
    'lxml>=3.3',
    'numpy>=1.9',
    'pandas>=1.0.0',
    'pyyaml>=5.3.1',
    'scipy>=0.13',
    'tqdm>=4.0',
    'joblib>=1.0.1',
] + [
    'bluepy>=2.1',
    'bluepysnap>=0.13.2,!=1.0.0,!=1.0.1',
    'libsonata>=0.1.6',
    'morphio>=3,<4',
    'voxcell>=3.1.1',
]

SUBCELLULAR_REQUIRES = [
    'attrs<20',   # to use entity-management<1.0, need to use old attrs
    'entity-management>=0.1.11,<1.0',
    'subcellular-querier>=0.0.3',
    'tables>=3.4',
]

setup(
    name='brainbuilder',
    author='NSE Team',
    author_email='bbp-ou-nse@groupes.epfl.ch',
    version=VERSION,
    description='BrainBuilder is a tool to define the cells that will compose a circuit.',
    long_description='BrainBuilder is a tool to define the cells that will compose a circuit.',
    url='https://bbpteam.epfl.ch/project/issues/projects/BRBLD/issues/',
    download_url='https://bbpteam.epfl.ch/repository/devpi/+search?query=name%3Abrainbuilder',
    license='BBP-internal-confidential',
    install_requires=BASE_REQUIRES,
    extras_require={
        'all': SUBCELLULAR_REQUIRES,
        'subcellular': SUBCELLULAR_REQUIRES,
        'reindex': [],  # for compatibility
    },
    packages=find_packages(),
    python_requires='>=3.7',
    entry_points={
      'console_scripts': [
          'brainbuilder=brainbuilder.app.__main__:main'
      ]
    }
)
