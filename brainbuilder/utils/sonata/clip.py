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


def morphologies(output: str | Path, circuit_path: str, population_name: str) -> None:
    """Copy all morphologies used by a single population into a single
    output directory.

    Unlike `copy_filtered_morphologies`, this function dumps the
    morphologies into one standalone folder instead of copying
    between two circuit objects.
    """

    output = Path(output)
    circuit = bluepysnap.Circuit(circuit_path)

    if population_name not in circuit.nodes.population_names:
        raise BrainBuilderError(f"{population_name} missing from {circuit.nodes.population_names}")

    population = circuit.nodes[population_name]
    population_config = population.config
    morph_paths = list(population.get(properties="morphology").unique())

    if "morphologies_dir" in population_config:
        source = Path(population_config["morphologies_dir"])
        dest = output / source.name
        missing = _copy_files_with_extension(source, dest, morph_paths, "swc")
        if missing:
            L.warning(_format_missing(missing))

    if "alternate_morphologies" in population_config:
        alt_morphs = population_config["alternate_morphologies"]
        for extension, name in EXTENSIONS_MAPPING.items():
            if name in alt_morphs:
                source = Path(alt_morphs[name])
                dest = output / source.name
                missing = _copy_files_with_extension(source, dest, morph_paths, extension)
                if missing:
                    L.warning(_format_missing(missing))

def copy_filtered_morphologies(old_circuit: bluepysnap.Circuit, new_circuit: bluepysnap.Circuit) -> None:
    """Copy the morphologies of `new_circuit` in the correct locations sourcing them from `old_circuit`.

    `new_circuit` knows:
        - what should be copied
        - where it should be copied
    `old_circuit` knows:
        - from where things need to be copied

    `old_circuit` is supposed to contain everything that is in `new_circuit` and with the same structure. 
    It is allowed to contain more. In other words,
    `new_circuit` is, in general, a filtered version of `old_circuit` with new morphology paths.

    Unlike `morphologies`, this function copies between two circuit objects using the `new_circuit` as
    master to decide what to copy rather than dumping all the morphologies in a single folder.
    """

    for pop_name in new_circuit.nodes:
        new_pop_config = new_circuit.nodes[pop_name].config
        if pop_name not in old_circuit.nodes:
            raise BrainBuilderError(f"{pop_name} missing from {old_circuit.nodes.population_names}. `new_circuit` and `old_circuit` should differ only in the paths.")
        old_pop_config = old_circuit.nodes[pop_name].config
        morph_paths = list(new_circuit.nodes[pop_name].get(properties="morphology").unique())

        if "morphologies_dir" in new_pop_config:
            dest = Path(new_pop_config["morphologies_dir"])
            if "morphologies_dir" not in old_pop_config:
                raise BrainBuilderError(f"`morphologies_dir` missing from {old_pop_config}. `new_circuit` and `old_circuit` should differ only in the paths.")
            source = Path(old_pop_config["morphologies_dir"])
            missing = _copy_files_with_extension(source, dest, morph_paths, "swc")
            if missing:
                L.warning(_format_missing(missing))
        
        if "alternate_morphologies" in new_pop_config:
            if "alternate_morphologies" not in old_pop_config:
                raise BrainBuilderError(f"`alternate_morphologies` missing from {old_pop_config}. `new_circuit` and `old_circuit` should differ only in the paths.")
            for extension, name in EXTENSIONS_MAPPING.items():
                if name in new_pop_config["alternate_morphologies"]:
                    dest = Path(new_pop_config["alternate_morphologies"][name])
                    if name not in old_pop_config["alternate_morphologies"]:
                        raise BrainBuilderError(f"`{name}` missing from {old_pop_config['alternate_morphologies']}. `new_circuit` and `old_circuit` should differ only in the paths.")
                    source = Path(old_pop_config["alternate_morphologies"][name])
                    missing = _copy_files_with_extension(source, dest, morph_paths, extension)
                    if missing:
                        L.warning(_format_missing(missing))
