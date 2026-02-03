import logging
from pathlib import Path

import bluepysnap
import h5py

from brainbuilder.utils import hdf5
from brainbuilder.utils.sonata import layout

L = logging.getLogger(__name__)


def repair_circuit(output, circuit):
    """
    Repair circuits by inferring missing attributes.
    """
    output = Path(output)

    if isinstance(circuit, (str, Path)):
        circuit = bluepysnap.Circuit(circuit)
    else:
        assert isinstance(circuit, bluepysnap.Circuit)

    _, edge_pop_to_paths = layout._gather_layout_from_networks(circuit.config["networks"])

    _repair_neuroglial_edge_file(output, circuit, edge_pop_to_paths)


def _repair_neuroglial_edge_file(output, circuit, edge_pop_to_paths):
    """
    Repair a neuroglial edge HDF5 file by copying its contents while excluding
    `synapse_id` and `synapse_population`, then creating an appendable
    `target_edge_id` dataset with the corresponding synapse edge population attribute.
    """
    chemical_candidates = [name for name, e in circuit.edges.items() if e.type == "chemical"]

    for edge_pop_name, edge in circuit.edges.items():
        if edge.type != "synapse_astrocyte":
            continue

        edge_path = edge.h5_filepath
        output_path = output / edge_pop_to_paths[edge_pop_name]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(edge_path, "r") as h5in:
            orig_group = h5in["edges"][edge_pop_name]

            if "target_edge_id" in orig_group:
                L.info(f"`{edge_pop_name}` already contains target_edge_id, skipping")
                continue

            group0 = orig_group["0"]
            if "synapse_id" not in group0:
                L.warning(f"`{edge_pop_name}` missing synapse_id, cannot repair.")
                continue

            if len(chemical_candidates) != 1:
                raise RuntimeError(
                    f"Cannot infer synapse_population for repair, candidates={chemical_candidates}"
                )
            syn_pop = chemical_candidates[0]

            # Exclude synapse_population and synapse_id
            exclude_paths = {
                f"edges/{edge_pop_name}/0/synapse_population",
                f"edges/{edge_pop_name}/0/synapse_id",
            }

            with h5py.File(output_path, "w") as h5out:
                # Copy everything except excluded paths
                hdf5.copy_h5_filtered(h5in, h5out, exclude_paths=exclude_paths)

                # Use appendable dataset for target_edge_id
                src_ds = h5in[f"edges/{edge_pop_name}/0/synapse_id"]
                dst_group = h5out[f"edges/{edge_pop_name}"]
                dst_ds = hdf5.create_appendable_dataset(
                    dst_group, "target_edge_id", dtype=src_ds.dtype
                )
                hdf5.append_to_dataset(dst_ds, src_ds[()])

                # Add attribute
                dst_ds.attrs["edge_population"] = syn_pop
