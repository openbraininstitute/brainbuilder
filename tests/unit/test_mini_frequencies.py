"""
Test assignment of mini frequencies to the circuit MVD3.
"""
from pathlib import Path

import pandas as pd
import pytest
from brainbuilder.app import cells
from brainbuilder.exceptions import BrainBuilderError

DATA_PATH = Path(Path(__file__).parent, "data")


def test_mini_frequencies_input():
    """
    Mini frequencies input data must be able to read two columns from a TSV
    file.
    1. exc_mini_frequency
    2. inh_mini_frequency

    and the layer information should be in the index.
    """
    mini_freqs = cells.load_mini_frequencies(DATA_PATH / "mini_frequencies.tsv")
    assert mini_freqs.index.name == "layer"
    assert "exc_mini_frequency" in mini_freqs.columns
    assert "inh_mini_frequency" in mini_freqs.columns


def test_mini_frequencies_assignment():
    """
    Mini frequencies must be assigned to cells.
    """
    cells_df = pd.read_csv(
        DATA_PATH / "pre_mini_frequency_assignment_cells.tsv",
        sep="\t",
        index_col="gid")

    mini_freqs = cells.load_mini_frequencies(DATA_PATH / "mini_frequencies.tsv")

    cells._assign_mini_frequencies(cells_df, mini_freqs)

    expected_df = pd.read_csv(
        DATA_PATH / "expected-cells-with-mini-frequencies.tsv",
        sep="\t",
        index_col="gid")

    pd.testing.assert_frame_equal(cells_df, expected_df)

def test_mini_frequencies_assignment_no_layer():
    """
    Mini frequencies cannot be assigned to cells with no layer.
    """
    cells_df = pd.read_csv(
        DATA_PATH / "pre_mini_frequency_assignment_cells_no_layer.tsv",
        sep="\t",
        index_col="gid")

    mini_freqs = cells.load_mini_frequencies(DATA_PATH / "mini_frequencies.tsv")

    with pytest.raises(BrainBuilderError, match="Cannot add mini frequencies to cells with no layer"):
        cells._assign_mini_frequencies(cells_df, mini_freqs)
