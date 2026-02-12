from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from brainbuilder.utils.sonata import repair_circuit


@pytest.fixture
def sample_h5(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as f:
        grp = f.require_group("edges/pop_neuroglial/0")
        grp.create_dataset("synapse_id", data=np.arange(5))
        grp.create_dataset("synapse_population", data=np.full(5, b"pop_chemical"))
        grp.create_dataset("other_ds", data=np.ones(5))
    return path

@pytest.fixture
def dummy_circuit(sample_h5):
    config = {"networks": {"nodes": {}, "edges": [{"edges_file": "pop_neuroglial.h5", "populations": ["pop_neuroglial"]}]}}
    edges = {
        "pop_chemical": SimpleNamespace(type="chemical", h5_filepath=sample_h5),
        "pop_neuroglial": SimpleNamespace(type="synapse_astrocyte", h5_filepath=sample_h5),
    }
    return SimpleNamespace(edges=edges, config=config)

def test_repair_neuroglial_edge_file(tmp_path, sample_h5, dummy_circuit):
    out_dir = tmp_path

    repair_circuit.repair_neuroglial_edge_file(out_dir, dummy_circuit)

    out_file = out_dir / "pop_neuroglial/pop_neuroglial.h5"
    assert out_file.exists()

    with h5py.File(out_file, "r") as f:
        grp = f["edges/pop_neuroglial/0"]

        # synapse_population must be removed
        assert "synapse_population" not in grp

        # synapse_id must exist and have attribute
        assert "synapse_id" in grp
        syn_id = grp["synapse_id"]
        np.testing.assert_array_equal(syn_id[()], np.arange(5))
        assert syn_id.attrs["edge_population"] == "pop_chemical"

        # other datasets must be preserved
        assert "other_ds" in grp
        np.testing.assert_array_equal(grp["other_ds"][()], np.ones(5))

def test_repair_neuroglial_edge_file_skips_if_already_repaired(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as f:
        grp = f.require_group("edges/pop_neuroglial/0")
        syn_id = grp.create_dataset("synapse_id", data=np.arange(3))
        syn_id.attrs["edge_population"] = "pop_chemical"
        grp.create_dataset("other_ds", data=np.ones(3))

    edges = {
        "pop_chemical": SimpleNamespace(type="chemical", h5_filepath=path),
        "pop_neuroglial": SimpleNamespace(type="synapse_astrocyte", h5_filepath=path),
    }
    config = {"networks": {"nodes": {}, "edges": [{"edges_file": "pop_neuroglial.h5", "populations": ["pop_neuroglial"]}]}}
    circuit = SimpleNamespace(edges=edges, config=config)


    repair_circuit.repair_neuroglial_edge_file(tmp_path, circuit)

    # Should skip entirely → no output file
    assert not (tmp_path / "pop_neuroglial/pop_neuroglial.h5").exists()

def test_repair_aborts_on_multiple_synapse_populations(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as f:
        grp = f.require_group("edges/pop_neuroglial/0")
        grp.create_dataset("synapse_id", data=np.arange(4))
        grp.create_dataset("synapse_population", data=np.array([1, 2, 1, 2]))

    edges = {
        "pop_chemical": SimpleNamespace(type="chemical", h5_filepath=path),
        "pop_neuroglial": SimpleNamespace(type="synapse_astrocyte", h5_filepath=path),
    }
    config = {"networks": {"nodes": {}, "edges": [{"edges_file": "pop_neuroglial.h5", "populations": ["pop_neuroglial"]}]}}
    circuit = SimpleNamespace(edges=edges, config=config)

    with pytest.raises(RuntimeError, match="multiple synapse populations"):
        repair_circuit.repair_neuroglial_edge_file(
            tmp_path, circuit
        )

def test_repair_fallback_on_empty_synapse_population(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as f:
        grp = f.require_group("edges/pop_neuroglial/0")
        grp.create_dataset("synapse_id", data=np.arange(3))
        grp.create_dataset("synapse_population", data=np.array([], dtype=int))

    edges = {
        "pop_chemical": SimpleNamespace(type="chemical", h5_filepath=path),
        "pop_neuroglial": SimpleNamespace(type="synapse_astrocyte", h5_filepath=path),
    }
    config = {"networks": {"nodes": {}, "edges": [{"edges_file": "out.h5", "populations": ["pop_neuroglial"]}]}}
    circuit = SimpleNamespace(edges=edges, config=config)

    # Should succeed and fallback to chemical candidate
    repair_circuit.repair_neuroglial_edge_file(tmp_path, circuit)

    out_file = tmp_path / "pop_neuroglial/out.h5"
    assert out_file.exists()

    with h5py.File(out_file, "r") as f:
        grp = f["edges/pop_neuroglial/0"]

        # synapse_population removed
        assert "synapse_population" not in grp

        # synapse_id exists and has edge_population attribute from chemical
        syn_id = grp["synapse_id"]
        np.testing.assert_array_equal(syn_id[()], np.arange(3))
        assert syn_id.attrs["edge_population"] == "pop_chemical"

def test_repair_aborts_with_multiple_chemical_candidates(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as f:
        grp = f.require_group("edges/pop_neuroglial/0")
        grp.create_dataset("synapse_id", data=np.arange(2))

    edges = {
        "chem_1": SimpleNamespace(type="chemical", h5_filepath=path),
        "chem_2": SimpleNamespace(type="chemical", h5_filepath=path),
        "pop_neuroglial": SimpleNamespace(type="synapse_astrocyte", h5_filepath=path),
    }
    config = {"networks": {"nodes": {}, "edges": [{"edges_file": "out.h5", "populations": ["pop_neuroglial"]}]}}
    circuit = SimpleNamespace(edges=edges, config=config)

    with pytest.raises(RuntimeError, match="chemical candidates"):
        repair_circuit.repair_neuroglial_edge_file(
            tmp_path, circuit
        )

