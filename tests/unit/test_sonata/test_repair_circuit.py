import pytest
import h5py
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from brainbuilder.utils.sonata import repair_circuit

@pytest.fixture
def sample_h5(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as f:
        grp = f.require_group("edges/pop_neuroglial/0")
        grp.create_dataset("synapse_id", data=np.arange(5))
        grp.create_dataset("synapse_population", data=np.arange(5) + 10)
        grp.create_dataset("other_ds", data=np.ones(5))
    return path

@pytest.fixture
def dummy_circuit(sample_h5):
    edges = {
        "pop_chemical": SimpleNamespace(type="chemical", h5_filepath=sample_h5),
        "pop_neuroglial": SimpleNamespace(type="synapse_astrocyte", h5_filepath=sample_h5),
    }
    return SimpleNamespace(edges=edges)

def test_repair_neuroglial_edge_file(tmp_path, sample_h5, dummy_circuit):
    edge_pop_to_paths = {"pop_neuroglial": Path("pop_neuroglial.h5")}
    out_dir = tmp_path

    repair_circuit._repair_neuroglial_edge_file(out_dir, dummy_circuit, edge_pop_to_paths)

    out_file = out_dir / "pop_neuroglial.h5"
    assert out_file.exists()

    with h5py.File(out_file, "r") as f:
        grp = f["edges/pop_neuroglial/0"]

        # Excluded datasets should not exist
        assert "synapse_id" not in grp
        assert "synapse_population" not in grp

        # Other dataset should be copied
        assert "other_ds" in grp
        np.testing.assert_array_equal(grp["other_ds"][()], np.ones(5))

        # target_edge_id exists and has correct data
        target_ds = f["edges/pop_neuroglial/target_edge_id"]
        np.testing.assert_array_equal(target_ds[()], np.arange(5))
        assert target_ds.attrs["edge_population"] == "pop_chemical"
