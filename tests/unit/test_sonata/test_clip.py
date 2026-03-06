# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
from shutil import copy

import bluepysnap
import h5py
import numpy as np
import pytest
import utils

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

def _check_copy_filtered_morphologies(old_path, new_path):
    """Test helper that prepares a filtered circuit and verifies that
    `copy_filtered_morphologies` copies only the morphologies still
    referenced in the new nodes file.
    """

    shutil.rmtree(new_path, ignore_errors=True)
    new_path.mkdir()

    old_circuit_path = old_path / "circuit_config.json"
    new_circuit_path = new_path / "new_circuit_config.json"
    old_nodes_path = old_path / "nodes.h5"
    new_nodes_path = new_path / "nodes.h5"
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
        assert (new_path / ext / f"0/0/0.{ext}").exists()
        assert (new_path / ext / f"1.{ext}").exists()

        assert (old_path / ext / f"2.{ext}").exists()
        # this was filtered out. Let's check if it is true
        assert not (new_path / ext / f"2.{ext}").exists()



def test_copy_filtered_morphologies(tmp_path):
    """Test that morphologies are correctly copied under various symlink
    and absolute path scenarios, including individual files, folders, and BASE_DIR rebasing.
    """
    old_path = tmp_path / "clip"
    new_path = tmp_path / "new_clip"
    shutil.copytree(DATA_PATH, old_path)

    # base
    _check_copy_filtered_morphologies(old_path=DATA_PATH, new_path=new_path)

    # Symlink in h5v1 folder
    utils.move_and_symlink(src=old_path / "h5/1.h5", dst_dir=tmp_path / "other")
    _check_copy_filtered_morphologies(old_path=DATA_PATH, new_path=new_path)
    utils.revert_move_and_symlink(symlink_path=old_path / "h5/1.h5")

    # Symlink the full h5v1 folder
    utils.move_and_symlink(src=old_path / "h5", dst_dir=tmp_path / "other")
    _check_copy_filtered_morphologies(old_path=DATA_PATH, new_path=new_path)
    utils.revert_move_and_symlink(symlink_path=old_path / "h5")

    # BASE_DIR is absolute
    utils.replace_json_values(old_path / "circuit_config.json", {"$BASE_DIR": str(old_path.resolve())})
    _check_copy_filtered_morphologies(old_path=DATA_PATH, new_path=new_path)

    # BASE_DIR and h5v1 are absolute
    utils.replace_json_values(old_path / "circuit_config.json", {"h5v1": str(old_path.resolve() / "h5")})
    _check_copy_filtered_morphologies(old_path=DATA_PATH, new_path=new_path)

    # BASE_DIR and h5v1 are absolute. Symlink in h5v1 folder
    utils.move_and_symlink(src=old_path / "h5/1.h5", dst_dir=tmp_path / "other")
    _check_copy_filtered_morphologies(old_path=DATA_PATH, new_path=new_path)
    utils.revert_move_and_symlink(symlink_path=old_path / "h5/1.h5")

    # BASE_DIR and h5v1 are absolute. Symlink the full h5v1 folder
    utils.move_and_symlink(src=old_path / "h5", dst_dir=tmp_path / "other")
    _check_copy_filtered_morphologies(old_path=DATA_PATH, new_path=new_path)
    utils.revert_move_and_symlink(symlink_path=old_path / "h5")

