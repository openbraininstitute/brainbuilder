# SPDX-License-Identifier: Apache-2.0
"""SONATA helpers to remove unnecessary (ie: clip) auxiliary files"""

import logging
import shutil
from pathlib import Path

import bluepysnap
from bluepysnap.morph import EXTENSIONS_MAPPING

from brainbuilder import BrainBuilderError

L = logging.getLogger(__name__)
MAX_MISSING_FILES_DISPLAY = 10


def _format_missing(missing, max_to_show=MAX_MISSING_FILES_DISPLAY):
    """truncate `missing` to `max_to_show`"""
    examples = missing[:max_to_show]
    if len(missing) > max_to_show:
        examples.append("...")
    filenames = "".join(f"\t{e}\n" for e in examples)
    return f"Missing {len(missing)} files:\n{filenames}"


def _copy_files_with_extension(source, dest, names, extension):
    """copy files w/ `names` and `extension` from `source` to `dest`"""
    L.info("Copying %s `%s` files: %s -> %s", len(names), extension, source, dest)
    missing = []
    extensions = [extension, extension.upper()]
    for name in names:
        target_dir = dest / Path(name).parent
        target_dir.mkdir(parents=True, exist_ok=True)

        for ext in extensions:
            path = source / f"{name}.{ext}"
            if path.exists():
                shutil.copy2(path, target_dir)
                break
        else:
            missing.append(name)

    return missing


def morphologies(
    output: str | Path | None,
    circuit: str | bluepysnap.Circuit,
    population_name: str,
    filtering_circuit: str | bluepysnap.Circuit | None = None,
):
    """copy only used morphologies to `output`

    The filtering is based on `filtering_circuit` if present
    If output is None, the new placement is provided by `filtering_circuit`.
        In that case `filtering_circuit` is required
    """

    if output is not None:
        output = Path(output)
    if isinstance(circuit, str) or isinstance(circuit, Path):
        circuit = bluepysnap.Circuit(circuit)

    if isinstance(filtering_circuit, str) or isinstance(circuit, Path):
        filtering_circuit = bluepysnap.Circuit(filtering_circuit)

    if population_name not in circuit.nodes.population_names:
        raise BrainBuilderError(f"{population_name} missing from {circuit.nodes.population_names}")

    population_config = circuit.nodes[population_name].config
    population = (
        circuit.nodes[population_name]
        if filtering_circuit is None
        else filtering_circuit.nodes[population_name]
    )
    morph_paths = list(population.get(properties="morphology").unique())

    if "morphologies_dir" in population_config:
        source = Path(population_config["morphologies_dir"])

        dest = (
            output / source.name
            if output is not None
            else Path(filtering_circuit.nodes[population_name].config["morphologies_dir"])
        )
        missing = _copy_files_with_extension(source, dest, morph_paths, "swc")
        if missing:
            L.warning(_format_missing(missing))

    if "alternate_morphologies" in population_config:
        alt_morphs = population_config["alternate_morphologies"]
        for extension, name in EXTENSIONS_MAPPING.items():
            if name in alt_morphs:
                source = Path(alt_morphs[name])
                dest = (
                    output / source.name
                    if output is not None
                    else Path(
                        filtering_circuit.nodes[population_name].config["alternate_morphologies"][
                            name
                        ]
                    )
                )

                missing = _copy_files_with_extension(source, dest, morph_paths, extension)
                if missing:
                    L.warning(_format_missing(missing))
