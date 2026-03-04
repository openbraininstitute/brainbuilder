# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from shutil import copy

import bluepysnap
import h5py
import numpy as np
import pytest

from brainbuilder import BrainBuilderError
from brainbuilder.utils.sonata import clip as test_module
from brainbuilder.utils.sonata.extract_subcircuit import rebase_config_file

DATA_PATH = (Path(__file__).parent / "../data/sonata/clip").resolve()


def test__format_missing():
    missing = [f"{r}.asc" for r in range(20)]
    ret = test_module._format_missing(missing, max_to_show=0)
    assert "Missing 20 files" in ret

    ret = test_module._format_missing(missing, max_to_show=3)
    assert "0.asc" in ret
    assert "4.asc" not in ret

    ret = test_module._format_missing(missing, max_to_show=10)
    assert "9.asc" in ret
    assert "10.asc" not in ret


def test__copy_files_with_extension(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir(parents=True, exist_ok=True)

    at_root = [str(r) for r in range(3)]
    subdir = [f"{r}/{r}/{r}" for r in range(3)]

    names = at_root + subdir
    for name in names:
        (source / Path(name).parent).mkdir(parents=True, exist_ok=True)
        open(source / f"{name}.asc", "w").close()

    names += ["missing0", "missing1"]

    missing = test_module._copy_files_with_extension(source, dest, names, extension="asc")

    assert missing == ["missing0", "missing1"]

    for ext in ("asc", "swc", "h5"):
        for name in names:
            assert (source / f"{name}.{ext}").exists


def test_morphologies(tmp_path):
    """Test that `morphologies` correctly copies all morphologies for a
    single population and raises an error for a missing population.
    Checks that the expected files exist in the output folder.
    """
    circuit_config = DATA_PATH / "circuit_config.json"

    with pytest.raises(BrainBuilderError):
        test_module.morphologies(tmp_path, circuit_config, "missing_population")

    test_module.morphologies(tmp_path, circuit_config, "A")

    for ext in ("asc", "swc", "h5"):
        assert (tmp_path / ext / f"0/0/0.{ext}").exists()
        assert (tmp_path / ext / f"1.{ext}").exists()
        assert (tmp_path / ext / f"2.{ext}").exists()

def _remove_elements_from_dataset(f, dataset_path: str, indexes: list[int]):
    """Remove specified elements from a dataset inside an open HDF5 file.

    Args:
        f: an open h5py.File object in "r+" mode
        dataset_path: HDF5 dataset path (e.g., "nodes/A/0/morphology")
        indexes: list of integer indexes to remove from the dataset
    """
    dset = f[dataset_path]
    data = dset[:]
    if len(data) == 0 or not indexes:
        return  # nothing to remove

    # Ensure indexes are valid
    indexes = sorted(set(i for i in indexes if 0 <= i < len(data)))
    if not indexes:
        return

    # Remove the elements
    data = np.delete(data, indexes)

    # Replace the dataset with the reduced data
    parent_group = f[dataset_path].parent
    del parent_group[dataset_path.split("/")[-1]]
    parent_group.create_dataset(dataset_path.split("/")[-1], data=data)

def test_copy_filtered_morphologies(tmp_path):
    """Test that `copy_filtered_morphologies` correctly filters and copies
    only the morphologies specified in `new_circuit`.

    Modifies the nodes.h5 dataset to simulate removed elements and
    checks that the corresponding files exist or are correctly filtered
    in the output folders.
    """
    old_circuit_path = DATA_PATH / "circuit_config.json"
    new_circuit_path = tmp_path / "new_circuit_config.json"
    old_nodes_path = DATA_PATH / "nodes.h5"
    new_nodes_path = tmp_path / "nodes.h5"
    copy(old_circuit_path, new_circuit_path)
    copy(old_nodes_path, new_nodes_path)

    # remove one element to do some real filtering. morphology `2` appears
    # at indexes 2 and 5
    ids = [2, 5]
    with h5py.File(new_nodes_path, "r+") as f:
        _remove_elements_from_dataset(f, "nodes/A/0/morphology", ids)
        _remove_elements_from_dataset(f,"nodes/A/0/model_type", ids)
        _remove_elements_from_dataset(f,"nodes/A/0/mtype", ids)
        _remove_elements_from_dataset(f,"nodes/A/node_type_id", ids)

    old_circuit = bluepysnap.Circuit(old_circuit_path)
    rebase_config_file(new_file_path=new_circuit_path, old_file_path=old_circuit_path)
    new_circuit = bluepysnap.Circuit(new_circuit_path)

    test_module.copy_filtered_morphologies(old_circuit=old_circuit, new_circuit=new_circuit)

    for ext in ("asc", "swc", "h5"):
        assert (tmp_path / ext / f"0/0/0.{ext}").exists()
        assert (tmp_path / ext / f"1.{ext}").exists()
        # this was filtered out. Let's check if it is true
        assert not (tmp_path / ext / f"2.{ext}").exists()






















# def create_test_circuit(tmp_path, population_name="A", morph_files=None, alt_morph_files=None):
#     morph_files = morph_files or ["0", "1", "2"]
#     alt_morph_files = alt_morph_files or {}

#     # Create directories
#     morph_dir = tmp_path / "morphologies"
#     morph_dir.mkdir(parents=True, exist_ok=True)
#     for f in morph_files:
#         (morph_dir / f"{f}.swc").write_text("dummy")

#     alt_morph_dirs = {}
#     for ext, name in alt_morph_files.items():
#         d = tmp_path / f"{name}_dir"
#         d.mkdir(parents=True, exist_ok=True)
#         for f in morph_files:
#             (d / f"{f}.{ext}").write_text("dummy")
#         alt_morph_dirs[name] = str(d)

#     class MockProperty:
#         def unique(self, *args, **kwargs):
#             return morph_files

#     class MockNode:
#         def __init__(self):
#             self.config = {"morphologies_dir": str(morph_dir), "alternate_morphologies": alt_morph_dirs}

#         def get(self, properties=None):
#             return MockProperty()

#     class MockNodes:
#         def __init__(self):
#             self.population_names = [population_name]
#             self._nodes = {population_name: MockNode()}

#         def __getitem__(self, key):
#             return self._nodes[key]

#         def __contains__(self, key):
#             return key in self._nodes

#     class MockCircuit:
#         def __init__(self):
#             self.nodes = MockNodes()

#     return MockCircuit()
















# def test_morphologies_with_filtering_circuit(tmp_path, monkeypatch):
#     """Test that morphologies are copied using a filtering circuit when output is None."""

#     # Create a source circuit
#     source_circuit = create_test_circuit(tmp_path)

#     # Create a filtering circuit with a different morphologies_dir
#     filtering_dir = tmp_path / "filtering_morphologies"
#     filtering_dir.mkdir()
#     filtering_circuit = create_test_circuit(tmp_path / "filtering", morph_files=["0", "1", "2"])
#     filtering_circuit.nodes["A"].config["morphologies_dir"] = str(filtering_dir)

#     # Patch bluepysnap.Circuit to return our mock circuits
#     monkeypatch.setattr(test_module, "bluepysnap", type("mock", (), {"Circuit": lambda x: source_circuit})())

#     # Patch _copy_files_with_extension to actually copy files to filtering_dir
#     def fake_copy_files(source, dest, names, extension):
#         dest.mkdir(parents=True, exist_ok=True)
#         for n in names:
#             (dest / f"{n}.{extension}").write_text("copied")
#         return []

#     monkeypatch.setattr(test_module, "_copy_files_with_extension", fake_copy_files)

#     # Run morphologies with output=None and filtering_circuit specified
#     test_module.morphologies(
#         output=None,
#         circuit=source_circuit,
#         population_name="A",
#         filtering_circuit=filtering_circuit,
#     )

#     # Assert files were copied into filtering_circuit morphologies_dir
#     for i in range(3):
#         assert (filtering_dir / f"{i}.swc").exists()

# def test_morphologies_error_if_population_missing(tmp_path):
#     """Test that missing population raises BrainBuilderError."""
#     circuit = create_test_circuit(tmp_path)
#     with pytest.raises(BrainBuilderError):
#         test_module.morphologies(output=tmp_path, circuit=circuit, population_name="missing_pop")

