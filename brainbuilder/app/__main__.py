"""
Collection of tools for circuit building.
"""

import logging
import click

from brainbuilder.app import cells, mvd3, nrn, sonata, syn2, targets, atlases
from brainbuilder.version import VERSION


def main():
    """ Collection of tools for circuit building """
    logging.basicConfig(level=logging.DEBUG)
    app = click.Group('brainbuilder', {
        'cells': cells.app,
        'mvd3': mvd3.app,
        'nrn': nrn.app,
        'sonata': sonata.app,
        'syn2': syn2.app,
        'targets': targets.app,
        'atlases': atlases.app,
    })
    app = click.version_option(VERSION)(app)
    app()


if __name__ == '__main__':
    main()
