import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from brainbuilder.utils.sonata import extract_subcircuit


@pytest.fixture
def tmp_json_file(tmp_path):
    """Create a temporary JSON file with absolute and relative paths as symlinks."""

    file_path = tmp_path / "test.json"
    real_dir = tmp_path / "real_files"
    real_dir.mkdir()

    # create the real files in another folder
    real_files = {
        "some_file.txt": real_dir / "some_file.txt",
        "nested.txt": real_dir / "nested.txt",
        "list1.txt": real_dir / "list1.txt",
    }
    for f in real_files.values():
        f.touch()

    # now create symlinks in tmp_path pointing to the real files
    (tmp_path / "some_file.txt").symlink_to(real_files["some_file.txt"])
    (tmp_path / "nested.txt").symlink_to(real_files["nested.txt"])
    (tmp_path / "list1.txt").symlink_to(real_files["list1.txt"])

    content = {
        "absolute_path": str(tmp_path / "some_file.txt"),
        "relative_path": "relative.txt",  # still a relative path (can create file separately)
        "nested": {"abs_nested": str(tmp_path / "nested.txt")},
        "list_paths": [str(tmp_path / "list1.txt"), "list2.txt"],
        "absolute_path_link": str(tmp_path / "some_file.txt"),
        "relative_path_link": "relative.txt",
    }

    # create the relative files in tmp_path
    (tmp_path / "relative.txt").touch()
    (tmp_path / "list2.txt").touch()

    file_path.write_text(json.dumps(content))
    return file_path, content, tmp_path

def test_copy_pop_hoc_files(tmp_path):
    """Copies only referenced .hoc files to destination directory."""
    source_dir = tmp_path / "source_hoc"
    dest_dir = tmp_path / "dest_hoc"
    source_dir.mkdir()

    hoc_name = "CellA.hoc"
    (source_dir / hoc_name).write_text("hoc content")

    # fake pop + circuit
    new_pop = SimpleNamespace(
        config={"biophysical_neuron_models_dir": str(dest_dir)},
        size=1,
        get=lambda properties=None: SimpleNamespace(unique=lambda: ["template:CellA"])
    )
    original_pop = SimpleNamespace(
        config={"biophysical_neuron_models_dir": str(source_dir)}
    )

    new_circuit = SimpleNamespace(nodes={"pop1": new_pop})
    original_circuit = SimpleNamespace(nodes={"pop1": original_pop})


    extract_subcircuit._copy_pop_hoc_files(
        new_circuit=new_circuit, original_circuit=original_circuit
    )

    assert (dest_dir / hoc_name).exists()

def test_copy_mod_files(tmp_path):
    """Copies mod directory if present next to circuit path."""
    # simulate circuit location
    circuit_file = tmp_path / "circuit.json"
    circuit_file.write_text("{}")

    # create mod folder next to it
    mod_dir = tmp_path / "mod"
    mod_dir.mkdir()
    (mod_dir / "channel.mod").write_text("mod content")

    output_root = tmp_path / "output"
    extract_subcircuit._copy_mod_files(str(circuit_file), str(output_root))

    assert (output_root / "mod" / "channel.mod").exists()



def test_rebase_config_file(tmp_json_file):
    """Test rebasing absolute paths in a JSON file on disk, including symlinks."""
    file_path, _content, tmp_base = tmp_json_file
    new_base = Path("$BASE_DIR")

    # Load JSON from file
    new_config = json.loads(file_path.read_text())

    # Rebase paths
    extract_subcircuit._recursive_rebase_paths(new_config, tmp_base, new_base)

    # Absolute paths (including symlinks) should now be under new_base
    assert new_config["absolute_path"] == str(new_base / "some_file.txt")
    assert new_config["absolute_path_link"] == str(new_base / "some_file.txt")
    assert new_config["relative_path"] == "relative.txt"  # relative paths unchanged
    assert new_config["relative_path_link"] == "relative.txt"
    assert new_config["nested"]["abs_nested"] == str(new_base / "nested.txt")
    assert new_config["list_paths"][0] == str(new_base / "list1.txt")
    assert new_config["list_paths"][1] == "list2.txt"


def test_recursive_rebase_paths(tmp_path):
    """Test rebasing absolute paths in a dict with files and symlinks."""
    old_base = tmp_path
    new_base = Path("$BASE_DIR")

    # Create real files and symlinks to them
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    files = ["a.txt", "c.txt", "d.txt"]
    for f in files:
        (real_dir / f).touch()

    # Symlinks in tmp_path pointing to real files
    for f in files:
        (tmp_path / f).symlink_to(real_dir / f)

    d = {
        "absolute_path": str(tmp_path / "a.txt"),
        "relative_path": "b.txt",
        "nested": {"abs_nested": str(tmp_path / "c.txt")},
        "list_paths": [str(tmp_path / "d.txt"), "e.txt"]
    }

    extract_subcircuit._recursive_rebase_paths(d, old_base, new_base)

    # Absolute paths now rebased under new_base; relative paths unchanged
    assert d["absolute_path"] == str(new_base / "a.txt")
    assert d["nested"]["abs_nested"] == str(new_base / "c.txt")
    assert d["list_paths"][0] == str(new_base / "d.txt")
    assert d["relative_path"] == "b.txt"
    assert d["list_paths"][1] == "e.txt"