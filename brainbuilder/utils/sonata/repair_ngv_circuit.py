from brainbuilder.utils import utils
from pathlib import Path
import bluepysnap
import logging
import numpy as np

import h5py

L = logging.getLogger(__name__)


def repair_ngv_circuit(output, circuit):
    output = Path(output)

    if isinstance(circuit, (str, Path)):
        circuit = bluepysnap.Circuit(circuit)
    else:
        assert isinstance(circuit, bluepysnap.Circuit)

    _, edge_pop_to_paths = utils.gather_layout_from_networks(circuit.config["networks"])

    _repair_neuroglial_edge_file(output, circuit, edge_pop_to_paths)


def _repair_neuroglial_edge_file(output, circuit, edge_pop_to_paths):
    chemical_candidates = [name for name, edge in circuit.edges.items() if edge.type == "chemical"]

    if len(chemical_candidates) != 1:
        raise RuntimeError(f"Cannot infer synapse_population, candidates={chemical_candidates}")

    syn_pop = chemical_candidates[0]

    for edge_pop_name, edge in circuit.edges.items():
        if edge.type != "synapse_astrocyte":
            continue

        edge_path = edge.h5_filepath
        output_path = output / edge_pop_to_paths[edge_pop_name]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(edge_path, "r") as h5in:
            orig_group = h5in["edges"][edge_pop_name]["0"]

            if "synapse_population" in orig_group:
                L.info(f"`{edge_pop_name}` already contains synapse_population, skipping")
                continue

            with h5py.File(output_path, "w") as h5out:
                for name in h5in:
                    h5in.copy(name, h5out)

                out_group = h5out["edges"][edge_pop_name]["0"]
                lib_group = h5out["edges"][edge_pop_name]["0"].require_group("@library")

                n = out_group["synapse_id"].shape[0]

                # indices into @library (all identical → 0)
                out_group.create_dataset(
                    "synapse_population",
                    shape=(n,),
                    dtype=np.int32,
                    data=np.zeros(n, dtype=np.int32),
                )

                # actual string stored once in @library
                lib_group.create_dataset(
                    "synapse_population",
                    shape=(1,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    data=np.array([syn_pop], dtype=object),
                )

        L.info("Neuroglial edge file repaired from %s to %s", edge_path, output_path)
