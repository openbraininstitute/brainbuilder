from utils import TEST_DATA_PATH, assert_h5_dirs_equal, assert_json_files_equal

from brainbuilder.utils.sonata import subsample as test_module


def test_subsample_circuit(tmp_path):
    output = tmp_path / "output"
    input_circuit_config = TEST_DATA_PATH / "sonata" / "subsample" / "circuit_config.json"
    expected_dir = TEST_DATA_PATH / "sonata" / "subsample" / "expected"
    test_module.subsample_circuit(
        output=output,
        delete=True,
        circuit_config=input_circuit_config,
        sampling_ratio=0.8,
        sampling_count=None,
        node_populations=None,
        seed=0,
    )

    assert_json_files_equal(output / "circuit_config.json", expected_dir / "circuit_config.json")
    assert_json_files_equal(output / "id_mapping.json", expected_dir / "id_mapping.json")
    assert_h5_dirs_equal(
        output / "networks" / "nodes" / "default",
        expected_dir / "networks" / "nodes" / "default",
    )
    assert_h5_dirs_equal(
        output / "networks" / "edges" / "default",
        expected_dir / "networks" / "edges" / "default",
    )
