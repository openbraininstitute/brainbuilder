from unittest.mock import Mock
import numpy as np
import tempfile

import voxcell
import pytest

from brainbuilder.app import cells as test_module
from brainbuilder.cell_positions import _get_cell_count


def test_load_density__dangerously_low_densities():
    """Test for very low densities where the float precision affects the total count."""

    shape = (10, 10, 10)
    voxel_dimensions = np.array([25, 25, 25])

    with tempfile.NamedTemporaryFile(suffix=".nrrd") as tfile:

        filepath = tfile.name

        raw = np.full(shape, dtype=np.float64, fill_value=2.04160691e02)

        raw[1, :, 1] = 8.36723867e10

        density = voxcell.VoxelData(raw=raw, voxel_dimensions=voxel_dimensions)
        density.save_nrrd(filepath)

        loaded_density = density.with_data(
            test_module._load_density(filepath, mask=np.ones(shape, int), atlas=None)
        )

        _, count = _get_cell_count(loaded_density, 1.0)

        assert count == 13073813
