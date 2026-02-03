import h5py
import numpy as np
import pytest

from brainbuilder.utils import hdf5


# ----------------------------------------------------------------------
#                               TESTS
# ----------------------------------------------------------------------

@pytest.fixture
def sample_h5(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("root_ds", data=[1, 2, 3])

        edges = f.create_group("edges")
        pop = edges.create_group("popA")
        g0 = pop.create_group("0")

        g0.create_dataset("a", data=np.arange(3))
        g0.create_dataset("b", data=np.arange(5))
        pop.create_dataset("top", data=[42])

    return path


def test_copy_all(sample_h5, tmp_path):
    out = tmp_path / "out.h5"

    with h5py.File(sample_h5, "r") as src, h5py.File(out, "w") as dst:
        hdf5.copy_h5_filtered(src, dst)

    with h5py.File(out, "r") as f:
        assert "root_ds" in f
        assert "edges/popA/0/a" in f
        assert "edges/popA/0/b" in f
        assert "edges/popA/top" in f


def test_exclude_paths(sample_h5, tmp_path):
    out = tmp_path / "out.h5"

    exclude = {
        "edges/popA/0/b",
        "root_ds",
    }

    with h5py.File(sample_h5, "r") as src, h5py.File(out, "w") as dst:
        hdf5.copy_h5_filtered(src, dst, exclude_paths=exclude)

    with h5py.File(out, "r") as f:
        assert "edges/popA/0/a" in f
        assert "edges/popA/0/b" not in f
        assert "root_ds" not in f

def test_dataset_content_preserved(sample_h5, tmp_path):
    out = tmp_path / "out.h5"

    with h5py.File(sample_h5, "r") as src, h5py.File(out, "w") as dst:
        hdf5.copy_h5_filtered(src, dst)

    with h5py.File(sample_h5, "r") as src, h5py.File(out, "r") as dst:
        np.testing.assert_array_equal(
            src["edges/popA/0/a"][:],
            dst["edges/popA/0/a"][:],
        )
