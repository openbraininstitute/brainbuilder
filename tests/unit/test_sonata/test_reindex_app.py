from pathlib import Path
import shutil
import tempfile
import h5py

from morph_tool import diff

from nose.tools import ok_
from numpy.testing import assert_allclose, assert_array_equal

TEST_DATA_PATH = Path(__file__).parent.parent / 'data'
DATA_PATH = TEST_DATA_PATH / 'sonata' / 'reindex'
MORPHS_PATH = DATA_PATH / 'morphs'


def update_morphologies(h5_morphs, nodes, population, output, edges):
    import subprocess
    subprocess.run(
        ['brainbuilder', 'sonata', 'update-morphologies', '-o', output, '--h5-morphs', h5_morphs],
        check=True, timeout=5 * 60)
    subprocess.run(
        ['brainbuilder', 'sonata', 'update-edge-population',
         '--h5-updates', output / 'h5_updates.json', '--nodes', nodes, '--', edges[0]],
        check=True, timeout=5 * 60)
    subprocess.run(
        ['brainbuilder', 'sonata', 'update-edge-pos',
         '--morph-path', output, '--nodes', nodes, '--', edges[0]],
        check=True, timeout=5 * 60)


def test__update_morphologies():
    temp_dir = tempfile.mkdtemp()
    temp_dir = Path(temp_dir)
    output_dir = temp_dir / 'output'
    temp_morph_dir = temp_dir / 'morph'
    temp_morph_dir.mkdir()
    try:
        shutil.copy(str(DATA_PATH / 'edges.h5'), temp_dir)
        shutil.copy(str(MORPHS_PATH / 'three_child_unmerged.h5'), str(temp_morph_dir))
        shutil.copy(str(MORPHS_PATH / 'two_child_unmerged.h5'), str(temp_morph_dir))
        # shutil.copy(str(MORPHS_PATH / 'three_child_unmerged.asc'), temp_dir)
        # shutil.copy(str(MORPHS_PATH / 'two_child_unmerged.asc'), temp_dir)
        edges_copy = temp_dir / 'edges.h5'
        update_morphologies(
            temp_morph_dir,
            DATA_PATH / 'nodes.h5',
            'default',
            output_dir,
            [edges_copy],
        )
        three_child_expected = MORPHS_PATH / 'three_child_merged.h5'
        three_child_updated = output_dir / 'three_child_unmerged.h5'
        with h5py.File(three_child_expected) as exp, h5py.File(three_child_updated) as act:
            assert_array_equal(exp['structure'][:], act['structure'][:])
            assert_allclose(exp['points'][:], act['points'][:])
        ok_(not diff(three_child_expected, three_child_updated))

        two_child_expected = MORPHS_PATH / 'two_child_merged.h5'
        two_child_updated = output_dir / 'two_child_unmerged.h5'
        with h5py.File(two_child_expected) as exp, h5py.File(two_child_updated) as act:
            assert_array_equal(exp['structure'][:], act['structure'][:])
            assert_allclose(exp['points'][:], act['points'][:])
        ok_(not diff(two_child_expected, two_child_updated))

        with h5py.File(edges_copy) as h5:
            grp = h5['/edges/default/0']
            assert_array_equal(grp['efferent_section_id'][:], [1,3,1,1, 2,3,3,1, 5,1,3,2])
            assert_allclose(grp['efferent_section_pos'][:], [2./3,4./7,0,1./6, .5,5.5/7,0,.25, 1,.4,.25,.75])
            assert_array_equal(grp['efferent_segment_id'][:], [3,1,0,0, 0,2,0,0, 1,1,0,1])
            assert_allclose(grp['efferent_segment_offset'][:], [1,1,0,2, 2,1.5,0,3, 3,0,1.5,1.5])

            assert_array_equal(grp['afferent_section_id'][:], [3,1,2,1, 6,3,4,1, 1,3,2,1])
            assert_allclose(grp['afferent_section_pos'][:], [5.5/7,3.5/12,3.5/4,1./6, 2./3,5./6,.5,0, 1,.5,.75,7./12])
            assert_array_equal(grp['afferent_segment_id'][:], [2,1,1,0, 1,1,0,0, 5,1,1,2])
            assert_allclose(grp['afferent_segment_offset'][:], [1.5,0.5,0.5,2, 1,2,1.5,0, 1,0.5,0,3])

    finally:
        shutil.rmtree(temp_dir)
