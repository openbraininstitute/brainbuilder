import logging
from pathlib import Path

import bluepysnap
import h5py

from brainbuilder.utils import hdf5
from brainbuilder.utils.sonata import _layout

L = logging.getLogger(__name__)


def repair_neuroglial_edge_file(output, circuit):
    """Repair a neuroglial edge HDF5 file by normalizing how the synapse edge population
    is stored.

    Rules:
    - If `synapse_id.attrs["edge_population"]` exists: file is already repaired → skip
    - Else, if `synapse_population` dataset exists:
        * if all values are identical → promote to attribute and drop dataset
        * if values differ → abort (multiple populations per file not supported)
    - Else, infer from chemical edge populations (must be exactly one)
    """

    output = Path(output)

    if isinstance(circuit, (str, Path)):
        circuit = bluepysnap.Circuit(circuit)

    _, edge_pop_to_paths = _layout.gather_layout_from_networks(circuit.config["networks"])

    chemical_candidates = [n for n, e in circuit.edges.items() if e.type == "chemical"]

    for edge_pop_name, edge in circuit.edges.items():
        if edge.type != "synapse_astrocyte":
            continue

        edge_path = edge.h5_filepath
        output_path = output / edge_pop_to_paths[edge_pop_name]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(edge_path, "r") as h5in:
            group0 = h5in["edges"][edge_pop_name]["0"]

            if "synapse_id" not in group0:
                L.warning(f"`{edge_pop_name}` missing synapse_id, cannot repair. Skipping")
                continue

            syn_id = group0["synapse_id"]

            # Already repaired
            if "edge_population" in syn_id.attrs:
                L.info(f"`{edge_pop_name}` already repaired. Skipping")
                continue

            syn_pop = None

            # Try synapse_population dataset
            if "synapse_population" in group0:
                sp = group0["synapse_population"][()]
                unique = set(sp.tolist())

                if len(unique) == 1:
                    syn_pop = unique.pop()
                elif len(unique) > 1:
                    raise RuntimeError(
                        f"`{edge_pop_name}` contains multiple synapse populations "
                        f"{sorted(unique)}. Multiple edge populations per single "
                        "neuro-glial edge filefile are no longer supported. "
                        "Split them into separate files."
                    )

            # Fallback: infer from chemical candidates
            if syn_pop is None:
                if len(chemical_candidates) != 1:
                    raise RuntimeError(
                        f"Cannot infer synapse_population for `{edge_pop_name}`, "
                        f"chemical candidates={chemical_candidates}"
                    )
                syn_pop = chemical_candidates[0]

            exclude_paths = {f"edges/{edge_pop_name}/0/synapse_population"}

            with h5py.File(output_path, "w") as h5out:
                # Copy everything except deprecated synapse_population
                hdf5.copy_h5_filtered(h5in, h5out, exclude_paths=exclude_paths)

                # Attach canonical attribute
                h5out[f"edges/{edge_pop_name}/0/synapse_id"].attrs["edge_population"] = syn_pop
