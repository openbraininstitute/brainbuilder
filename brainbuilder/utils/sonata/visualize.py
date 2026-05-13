# SPDX-License-Identifier: Apache-2.0
"""Visualize a SONATA circuit as a graph with populations as clusters."""

import json
from collections import Counter
from enum import StrEnum
from pathlib import Path

import bluepysnap
import h5py

# Populations with more nodes than this threshold are shown as a single
# cluster node with population-level edges instead of individual nodes.
_MAX_NODES_DETAILED = 10


class PopulationType(StrEnum):
    """Classification of a node population for visualization purposes."""

    BIOPHYSICAL = "biophysical"
    VIRTUAL = "virtual"
    EXTERNAL = "external"

    @property
    def color(self) -> str:
        return {
            PopulationType.BIOPHYSICAL: "lightyellow",
            PopulationType.VIRTUAL: "lightblue",
            PopulationType.EXTERNAL: "lightsalmon",
        }.get(self, "white")

    @classmethod
    def from_population(cls, pop_name: str, pop_type: str | None) -> "PopulationType":
        if pop_name.startswith("external_"):
            return cls.EXTERNAL
        if pop_type == "virtual":
            return cls.VIRTUAL
        return cls.BIOPHYSICAL


def _load_id_mapping(circuit_config_path):
    """Load id_mapping from circuit provenance if available.

    Returns:
        dict: pop_name -> list of parent_ids, or None if no mapping exists.
    """
    config = json.loads(Path(circuit_config_path).read_text())
    mapping_path = config.get("components", {}).get("provenance", {}).get("id_mapping")
    if not mapping_path:
        return None

    mapping_file = Path(circuit_config_path).parent / mapping_path
    if not mapping_file.exists():
        return None

    mapping = json.loads(mapping_file.read_text())
    return mapping


def draw_circuit(
    circuit_config_path, output_path=None, max_nodes_detailed=_MAX_NODES_DETAILED, title=None
):
    """Draw a SONATA circuit using graphviz with populations as clusters.

    For small populations (<=max_nodes_detailed), individual nodes and edges
    are shown. For large populations, a single summary node is shown with
    population-level edges. Duplicate edges are collapsed with a count label.

    If an id_mapping exists in provenance, node labels show the parent (original)
    IDs instead of the local IDs. External populations get a distinct color.

    Args:
        circuit_config_path: Path to circuit_config.json.
        output_path: If provided, save the rendered image to this path.
            Otherwise, render to a temp file and open it.
        max_nodes_detailed: Populations with more nodes than this are shown
            as a single summary node.

    Requires:
        pip install brainbuilder[viz]
        System graphviz: brew install graphviz (macOS) / apt install graphviz (Linux)
    """
    try:
        import graphviz
    except ImportError as e:
        raise ImportError(
            "graphviz Python package is required for visualization. "
            "Install with: pip install brainbuilder[viz]\n"
            "Also requires system graphviz: brew install graphviz"
        ) from e

    circuit = bluepysnap.Circuit(str(circuit_config_path))
    id_mapping = _load_id_mapping(circuit_config_path)

    dot = graphviz.Digraph("circuit", format="png")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="circle", fontsize="9", width="0.3", height="0.3")
    if title:
        dot.attr(label=title, labelloc="t", fontsize="14")

    detailed_pops = set()

    for pop_name, pop in circuit.nodes.items():
        pop_type = PopulationType.from_population(pop_name, pop.type)

        # Get original IDs for labels if mapping exists
        parent_ids = None
        if id_mapping and pop_name in id_mapping:
            entry = id_mapping[pop_name]
            parent_ids = entry.get("original_id", entry.get("parent_id"))

        if pop.size <= max_nodes_detailed:
            detailed_pops.add(pop_name)
            with dot.subgraph(name=f"cluster_{pop_name}") as sub:
                sub.attr(
                    label=f"{pop_name} ({pop_type}, {pop.size})",
                    style="filled",
                    color=pop_type.color,
                )
                prev = None
                for i in range(pop.size):
                    label = str(parent_ids[i]) if parent_ids else str(i)
                    sub.node(f"{pop_name}__{i}", label=f"<<B>{label}</B>>")
                    if prev is not None:
                        sub.edge(prev, f"{pop_name}__{i}", style="invis", weight="10")
                    prev = f"{pop_name}__{i}"
        else:
            dot.node(
                f"{pop_name}__summary",
                label=f"{pop_name}\n({pop_type}, {pop.size})",
                shape="box",
                style="filled",
                fillcolor=pop_type.color,
            )

    # Edges — group duplicates and show count
    for edge_name, edge in circuit.edges.items():
        src_name = edge.source.name
        tgt_name = edge.target.name
        src_detailed = src_name in detailed_pops
        tgt_detailed = tgt_name in detailed_pops

        with h5py.File(edge.h5_filepath, "r") as h5:
            sgids = h5[f"edges/{edge_name}/source_node_id"][:]
            tgids = h5[f"edges/{edge_name}/target_node_id"][:]

        if src_detailed and tgt_detailed:
            edge_counts = Counter(zip(sgids.tolist(), tgids.tolist()))
            for (s, t), count in edge_counts.items():
                attrs = {}
                if count > 1:
                    attrs["label"] = str(count)
                    attrs["fontsize"] = "8"
                dot.edge(f"{src_name}__{s}", f"{tgt_name}__{t}", **attrs)
        else:
            src_node = f"{src_name}__summary" if not src_detailed else f"{src_name}__{sgids[0]}"
            tgt_node = f"{tgt_name}__summary" if not tgt_detailed else f"{tgt_name}__{tgids[0]}"
            dot.edge(src_node, tgt_node, label=str(len(sgids)), fontsize="8")

    if output_path:
        dot.render(outfile=output_path, cleanup=True)
    else:
        import tempfile

        filename = title.replace(" ", "_") if title else "circuit"
        filepath = Path(tempfile.gettempdir()) / filename
        dot.render(filename=str(filepath), view=True, cleanup=True)
