'''test positions_and_orientations'''
import h5py
import yaml

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner

from voxcell import VoxelData  # type: ignore

import brainbuilder.app.cells as tested


def create_density_configuration():
    config = {
        'inputDensityVolumePath': {
            'inhibitory neuron': "inhibitory_neuron_density.nrrd",
            'excitatory neuron': "excitatory_neuron_density.nrrd",
            'oligodendrocyte': "oligodendrocyte_density.nrrd",
            'astrocyte': "astrocyte_density.nrrd",
            'microglia': "microglia_density.nrrd"
        }
    }

    return config


def create_input():
    input_ = {
            'annotation': np.array(
                [[[512, 512, 1143]], [[512, 512, 1143]], [[477, 56, 485]],]
            ),
            'orientation': np.array([
                [[
                    [1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0]
                ]],
                [[
                    [0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0],
                    [1, 0, 0, 1],
                    [1, 0, 0, 0]
                ]],
                [[
                    [0, 0, 0, 1],
                    [0.0, 1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
                    [1, 0, 0, 0]
                ]]
            ]),
            'inhibitory neuron':
                np.array([[[0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]],]),
            'excitatory neuron':
                np.array([[[0.0, 1.0, 0.0]], [[9.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]],]),
            'astrocyte':
                np.array([[[0.0, 1.0, 5.0]], [[1.0, 4.0, 5.0]], [[0.0, 1.0, 0.0]],]),
            'microglia':
                np.array([[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]],]),
            'oligodendrocyte':
                np.array([[[1.0, 1.0, 0.0]], [[2.0, 1.0, 4.0]], [[0.0, 0.0, 0.0]],]),
        }

    return input_



def test_positions_and_orientations_valid_input():
    voxel_dimensions = [25] * 3  # a voxel of size 25um x 25um x 25um
    input_ = create_input()
    config = create_density_configuration()
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('config.yaml', 'w') as out:
            yaml.dump(config, out)
        for cell_type, path in config['inputDensityVolumePath'].items():
            # the input densities are expressed in number of cells per voxel
            VoxelData(input_[cell_type] * (1e9 / 25 ** 3), voxel_dimensions=voxel_dimensions)\
                .save_nrrd(path)
        for input_voxel_data in ['annotation', 'orientation']:
            VoxelData(input_[input_voxel_data], voxel_dimensions=voxel_dimensions)\
                .save_nrrd(input_voxel_data + '.nrrd')
        result = runner.invoke(
            tested.positions_and_orientations,
            [
                '--annotation-path',
                'annotation.nrrd',
                '--orientation-path',
                'orientation.nrrd',
                '--config-path',
                'config.yaml',
                '--output-path',
                'positions_and_orientations.h5'
            ],
        )
        assert result.exit_code == 0
        cell_collections = h5py.File('positions_and_orientations.h5', 'r')
        npt.assert_array_almost_equal(
            cell_collections.get('/nodes/astrocyte/0/orientation_x')[()],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.57735027, 0.70710678,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        npt.assert_array_almost_equal(
            cell_collections.get('/nodes/astrocyte/0/y')[()],
            [19.45391877, 19.97896411, 2.95686065, 23.61672293, 6.6138903, 14.21084872,
            15.30239307, 17.04550748, 17.4407799, 16.76594674, 7.88570877, 10.96503784,
            5.2219189, 6.33229006, 3.97423959, 3.45457378, 20.52483075]
        )
        npt.assert_array_equal(
            cell_collections.get('/nodes/astrocyte/0/region_id')[()],
            [512, 1143, 512, 512, 512, 512, 512, 1143, 56, 512, 1143, 512,
            512, 1143, 1143, 1143, 512]
        )


def test_positions_and_orientations_invalid_input():
    config = create_density_configuration()
    input_ = create_input()
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('config.yaml', 'w') as out:
            yaml.dump(config, out)
        for cell_type, path in config['inputDensityVolumePath'].items():
            # the input densities are expressed in number of cells per voxel
            VoxelData(input_[cell_type] * (1e9 / 25 ** 3), voxel_dimensions=[25] * 3)\
                .save_nrrd(path)
        for input_voxel_data in ['annotation', 'orientation']:
            # Intentional mismatch of voxel dimensions: 10um != 25um
            VoxelData(input_[input_voxel_data], voxel_dimensions=[10] * 3)\
                .save_nrrd(input_voxel_data + '.nrrd')
        result = runner.invoke(
            tested.positions_and_orientations,
            [
                '--annotation-path',
                'annotation.nrrd',
                '--orientation-path',
                'orientation.nrrd',
                '--config-path',
                'config.yaml',
                '--output-path',
                'positions_and_orientations.h5'
            ],
        )
        assert result.exit_code == 1
        assert 'different voxel dimensions' in str(result.exception)
