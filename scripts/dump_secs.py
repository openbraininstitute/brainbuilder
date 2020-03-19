#!/usr/bin/env python3
'''
For hippocampus morphologies:
module purge && module load unstable neurodamus-hippocampus/0.4 brainbuilder

Need to have neurodamus-hippocampus/ so template can load.
'''

from pathlib import Path
import collections
import json
import os
import sys

import click
import neuron

h = neuron.h

h.nrn_load_dll(os.environ['NRNMECH_LIB_PATH'])

DEFAULT_TEMPLATE = ('/gpfs/bbp.cscs.ch/project/proj42/entities/'
                    'emodels/20190402/hoc/CA1_int_cNAC_971114B_2019032915460.hoc')


def get_nseg_map(hoc_template, morph_path):
    '''load `morph_path` using `hoc_template` in NEURON, dump information about the sections'''
    h.load_file(str(hoc_template))
    tpl = getattr(h, hoc_template.stem)
    cell = tpl(0, str(morph_path.parent), str(morph_path.name))

    ret = collections.defaultdict(dict)
    for sec in cell.all:
        n3d = sec.n3d()
        ret[str(sec)] = {'nseg': sec.nseg,
                         'n3d': n3d,
                         'arc3d': ['%.5f' % sec.arc3d(i) for i in range(n3d)],
                         'x3d': ['%.5f' % sec.x3d(i) for i in range(n3d)],
                         'y3d': ['%.5f' % sec.y3d(i) for i in range(n3d)],
                         'z3d': ['%.5f' % sec.z3d(i) for i in range(n3d)],
                         'diam3d': ['%.5f' % sec.diam3d(i) for i in range(n3d)],
                         }
    return ret


@click.command()
@click.option('--hoc-template', default=DEFAULT_TEMPLATE,
              help='path to hoc template to use for loading')
@click.option('--output', required=True,
              help='name/path of output.json file')
@click.argument('morph_path')
def dump_morphology(hoc_template, morph_path, output):
    nseg_map = get_nseg_map(Path(hoc_template), Path(morph_path))
    with open(output, 'w') as fd:
        json.dump(nseg_map, fd, indent=2, sort_keys=True)
    print('Wrote: %s' % output


if __name__ == '__main__':
    dump_morphology()
