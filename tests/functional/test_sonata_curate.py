"""Application of `curate` functionality to existing Sonata circuits."""
import shutil
from pathlib import Path

from bluepysnap.circuit_validation import Error, validate

from brainbuilder.utils.sonata import curate
from brainbuilder.utils.sonata.write_config import write_network_config


def test_hippocampus(tmp_path):
    """Example of curating a Hippocampus circuit"""
    circuit_path = Path("/gpfs/bbp.cscs.ch/project/proj42/circuits/CA1.O0/20191017/")
    proj_edges_file = circuit_path / "projections" / "v3.2k" / "O0_ca1_20191017_sorted.sonata"
    edges_file = circuit_path / "sonata" / "networks" / "edges" / "functional" / "All" / "edges.h5"
    nodes_file = circuit_path / "sonata" / "networks" / "nodes" / "All" / "nodes.h5"
    original_dir = tmp_path / "original"
    original_dir.mkdir()
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    # we use `copyfile` because the original file has restricted permissions, we don't
    # want to preserve them.
    shutil.copyfile(edges_file, original_dir / edges_file.name)
    edges_file = original_dir / edges_file.name
    shutil.copyfile(nodes_file, original_dir / nodes_file.name)
    nodes_file = original_dir / nodes_file.name
    shutil.copyfile(proj_edges_file, original_dir / proj_edges_file.name)
    proj_edges_file = original_dir / proj_edges_file.name

    target_nodes_name = "hippocampus_neurons"
    source_nodes_name = "hippocampus_projections"
    syn_type = "chemical"

    curate.rename_node_population(nodes_file, target_nodes_name)
    curate.set_group_attribute(
        nodes_file, "nodes", target_nodes_name, "0", "model_type", "biophysical", True
    )
    curate.rewire_edge_population(edges_file, nodes_file, nodes_file, syn_type)
    curate.add_edge_type_id(edges_file, curate.get_population_name(edges_file))

    proj_source_nodes_file = curate.create_projection_source_nodes(
        proj_edges_file, original_dir, source_nodes_name, fix_offset=True
    )
    start, _ = curate.get_source_nodes_range(proj_edges_file, edge_population_name='default')
    curate.correct_source_nodes_offset(proj_edges_file, edge_population_name='default', offset=start)

    curate.rewire_edge_population(proj_edges_file, proj_source_nodes_file, nodes_file, syn_type)

    curate.merge_h5_files([nodes_file, proj_source_nodes_file], "nodes", curated_dir / "nodes.h5")
    curate.merge_h5_files([edges_file, proj_edges_file], "edges", curated_dir / "edges.h5")

    sonata_config_file = curated_dir / "circuit_config.json"
    curated_dir = curated_dir.resolve()
    write_network_config(
        base_dir="/",
        morph_dir="/gpfs/bbp.cscs.ch/project/proj42/entities/morphologies/20180417/",
        emodel_dir="/gpfs/bbp.cscs.ch/project/proj42/entities/emodels/20190402/hoc",
        nodes_dir=curated_dir,
        nodes=[
            {
                "nodes_file": "nodes.h5",
                "populations": {
                    "hippocampus_neurons": {"type": "biophysical"},
                    "hippocampus_projections": {"type": "virtual"},
                },
            }
        ],
        node_sets="",
        edges_dir=curated_dir,
        edges_suffix="",
        edges=[
            {
                "edges_file": "edges.h5",
                "populations": {
                    "hippocampus_neurons__hippocampus_neurons__chemical": {"type": "chemical"},
                    "hippocampus_projections__hippocampus_neurons__chemical": {"type": "chemical"},
                },
            }
        ],
        output_path=sonata_config_file,
    )
    errors = validate(str(sonata_config_file), skip_slow=False)
    errors = [str(err) for err in errors if err.level == Error.FATAL]
    assert errors == []
