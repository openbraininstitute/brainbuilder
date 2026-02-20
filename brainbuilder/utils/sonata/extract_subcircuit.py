from pathlib import Path
from brainbuilder import utils
import bluepysnap
import logging
import shutil

from brainbuilder.utils.sonata import split_population, clip

L = logging.getLogger(__name__)


def _recursive_rebase_paths(config, old_base: Path, new_base: Path) -> None:
    """Recursively replace absolute paths under old_base with paths relative to new_base."""
    if isinstance(config, dict):
        for key, value in config.items():
            config[key] = _recursive_rebase_paths(value, old_base, new_base)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            config[i] = _recursive_rebase_paths(item, old_base, new_base)
    elif isinstance(config, str):
        path = Path(config)

        # only process paths that exist in the old config
        if path.is_absolute():
            path_resolved = path.resolve()
            try:
                return str(new_base / path_resolved.relative_to(old_base))
            except ValueError:
                return config
        return config
    return config


def rebase_config_file(new_file_path: str | Path, old_file_path: str | Path) -> None:
    old_file_path = Path(old_file_path)
    new_file_path = Path(new_file_path)

    old_config = utils.load_json(old_file_path)
    new_config = utils.load_json(new_file_path)

    old_base = Path(old_config.get("manifest", {}).get("$BASE_DIR", "."))
    old_base = (old_file_path.parent / old_base).resolve()
    new_base = Path("$BASE_DIR")

    _recursive_rebase_paths(new_config, old_base, new_base)

    utils.dump_json(new_file_path, new_config)


def _copy_pop_hoc_files(
    pop_name: str, pop: bluepysnap.nodes.NodePopulation, original_circuit: bluepysnap.Circuit
) -> None:
    """Copy only the biophysical neuron model (.hoc) files actually used by a population."""
    og_pop = original_circuit.nodes[pop_name]
    if "biophysical_neuron_models_dir" not in pop.config:
        return
    hoc_file_list = [
        _hoc.split(":")[-1] + ".hoc" for _hoc in pop.get(properties="model_template").unique()
    ]
    L.info(
        f"Copying {len(hoc_file_list)} biophysical neuron models (.hoc) for"
        f" population '{pop_name}' ({pop.size})"
    )

    source_dir = og_pop.config["biophysical_neuron_models_dir"]
    dest_dir = pop.config["biophysical_neuron_models_dir"]
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    for _hoc_file in hoc_file_list:
        src_file = Path(source_dir) / _hoc_file
        dest_file = Path(dest_dir) / _hoc_file
        if not Path(src_file).exists():
            raise ValueError(f"ERROR: HOC file '{src_file}' missing!")
        if not Path(dest_file).exists():
            # Copy only, if not yet existing (could happen for shared hoc files
            # among populations)
            shutil.copyfile(src_file, dest_file)


def _copy_mod_files(circuit_path: str, output_root: str) -> None:
    """Copy NEURON mod files from the original circuit, if present."""
    mod_folder = "mod"
    source_dir = Path(circuit_path).parent / mod_folder
    if Path(source_dir).exists():
        L.info("Copying mod files")
        dest_dir = Path(output_root) / mod_folder
        shutil.copytree(source_dir, dest_dir)
    else:
        L.info("No mod files to copy: skip")


def extract_subcircuit(
    output: str | Path,
    node_set_name: str,
    circuit_path: str | Path,
    do_virtual: bool,
    create_external: bool,
    list_of_virtual_sources_to_ignore: tuple[str] = (),
):
    """Extract a subcircuit and copy all required assets (morphologies, HOC, mod files)."""
    original_circuit = split_population.split_subcircuit(
        output=output,
        node_set_name=node_set_name,
        circuit=circuit_path,
        do_virtual=do_virtual,
        create_external=create_external,
        list_of_virtual_sources_to_ignore=list_of_virtual_sources_to_ignore,
    )
    new_circuit_path = Path(output) / "circuit_config.json"
    rebase_config_file(new_file_path=new_circuit_path, old_file_path=circuit_path)

    new_circuit = bluepysnap.Circuit(new_circuit_path)
    for pop_name, pop in new_circuit.nodes.items():
        clip.morphologies(
            None, circuit=original_circuit, population_name=pop_name, filtering_circuit=new_circuit
        )
        _copy_pop_hoc_files(pop_name=pop_name, pop=pop, original_circuit=original_circuit)
    _copy_mod_files(circuit_path=circuit_path, output_root=output)

    L.info("Extraction DONE")
