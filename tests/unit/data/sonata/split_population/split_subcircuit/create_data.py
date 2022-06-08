'''
`lhs > rhs` is an edge from lhs to rhs

Have 3 populations, to test that a nodeset that address multiple one

 nodeA  nodeB  nodeC   <- population
 0a  >   0a       0a         ('A', 'B', 0, 0)
 1b  <   1b       1b         ('B', 'A', 1, 1)
 2a      2a   >   2a         ('B', 'C', 2, 2)
 3b      3b   <   3b         ('C', 'B', 3, 3)
 4a      4a       4a  <      ('A', 'C', 4, 4)
 4a      4a       4a  >      ('C', 'A', 4, 4)
 ^^
 |+ a/b are 'mtypes'
 + Node ID

After keeping only mtypes of 'a' type;

nodeA  nodeB  nodeC
0a  >   0a       0a                    ('A', 'B', 0, 0)
2a      2a   >   2a      Renumbered -> ('B', 'C', 1, 1)
4a      4a       4a  <                 ('A', 'C', 2, 2)

Note: Since nodes are being removed, only node IDs 0/2/4 will be kept, and they need to be renumbered
'''

import libsonata
import h5py
import numpy as np
from collections import namedtuple
Edge = namedtuple('Edge', 'src, tgt, sgid, tgid')

def make_edges(edge_file_name, edges):
    with h5py.File(edge_file_name, 'w') as h5:
        for e in edges:
            pop_name = f'{e.src}__{e.tgt}'
            ds = h5.create_dataset(f'/edges/{pop_name}/source_node_id', data=np.array(e.sgid, dtype=int))
            ds.attrs['node_population'] = e.src
            ds = h5.create_dataset(f'/edges/{pop_name}/target_node_id', data=np.array(e.tgid, dtype=int))
            ds.attrs['node_population'] = e.tgt
            h5.create_dataset(f'/edges/{pop_name}/0/delay', data=np.array([0.5] * len(e.tgid), dtype=float))
            h5.create_dataset(f'/edges/{pop_name}/edge_type_id', data=np.array([-1] * len(e.tgid), dtype=int))

    for e in edges:
        pop_name = f'{e.src}__{e.tgt}'
        libsonata.EdgePopulation.write_indices(
            edge_file_name, pop_name, source_node_count=10, target_node_count=10
        )


with h5py.File('nodes.h5', 'w') as h5:
    h5.create_dataset('/nodes/A/0/mtype', data=['a', 'b', 'a', 'b', 'a', 'b'])
    h5.create_dataset('/nodes/A/0/model_type', data=['biophysical']*6)
    h5.create_dataset('/nodes/A/node_type_id', data=[1, 1, 1, 1, 1, 1, ])
    h5.create_dataset('/nodes/B/0/mtype', data=['a', 'b', 'a', 'b', 'a', 'b'])
    h5.create_dataset('/nodes/B/0/model_type', data=['biophysical']*6)
    h5.create_dataset('/nodes/B/node_type_id', data=[1, 1, 1, 1, 1, 1, ])
    h5.create_dataset('/nodes/C/0/mtype', data=['a', 'b', 'a', 'b', 'a', 'b'])
    h5.create_dataset('/nodes/C/0/model_type', data=['biophysical']*6)
    h5.create_dataset('/nodes/C/node_type_id', data=[1, 1, 1, 1, 1, 1, ])

edges = (
    Edge('A', 'B', [0], [0]),
    Edge('B', 'A', [1], [1]),
    Edge('B', 'C', [2], [2]),
    Edge('C', 'B', [3], [3]),
    Edge('A', 'C', [4], [4]),
    Edge('C', 'A', [4], [4]),
    )

make_edges('edges.h5', edges)

'''
For the virtual nodes, two separate files, with 2 populations V1, and V2;
V1 innervates populations A, and B, which V2 innervates C

nodeV1  A  B                     nodeV1  A  B
0      >0                        0      >0
1         >1        --- keep -->
2         >0                     2         >0
3      >0                        3      >0

keep ->

nodeV2  C
0      >2
'''

with h5py.File('virtual_nodes_V1.h5', 'w') as h5:
    h5.create_dataset('/nodes/V1/0/model_type', data=['virtual',]*4)
    h5.create_dataset('/nodes/V1/node_type_id', data=[1, 1, 1, 1, ])

edges = (
    Edge('V1', 'A', [0, 3], [0, 0]),
    Edge('V1', 'B', [1, 2], [1, 0]),
    )
make_edges('virtual_edges_V1.h5', edges)

edges = (
    Edge('V2', 'C', [0], [2]),
    )
make_edges('virtual_edges_V2.h5', edges)
with h5py.File('virtual_nodes_V2.h5', 'w') as h5:
    h5.create_dataset('/nodes/V2/0/model_type', data=['virtual',]*1)
    h5.create_dataset('/nodes/V2/node_type_id', data=[1, ])
