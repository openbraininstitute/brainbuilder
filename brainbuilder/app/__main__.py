"""
Collection of tools for circuit building.
"""

import logging
import click

from brainbuilder.app import astrocytes, cells, mvd3, nrn, sonata, subcellular, targets
from brainbuilder.version import VERSION


def main():
    """ Collection of tools for circuit building """
    logging.basicConfig(level=logging.INFO)
    app = click.Group('brainbuilder', {
        'astrocytes': astrocytes.app,
        'cells': cells.app,
        'mvd3': mvd3.app,
        'nrn': nrn.app,
        'sonata': sonata.app,
        'subcellular': subcellular.app,
        'targets': targets.app,
    })
    app = click.version_option(VERSION)(app)
    app()


if __name__ == '__main__':
    main()
