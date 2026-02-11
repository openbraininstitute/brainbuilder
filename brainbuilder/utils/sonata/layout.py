from pathlib import Path


def extract_population_paths(key, networks):
    """extract populations from `network_base`; return dictionary with their file path"""
    key_name = f"{key}_file"
    ret = {}
    for stanza in networks[key]:
        filename = Path(stanza[key_name]).name
        if len(stanza["populations"]) == 1:
            population = next(iter(stanza["populations"]))
            ret[population] = str(Path(population) / filename)
        else:
            # multiple populations; need to group them into the same file
            base_path = Path(stanza[key_name]).parent.name
            for population in stanza["populations"]:
                ret[population] = str(Path(base_path) / filename)
    return ret


def gather_layout_from_networks(networks):
    """find the layout of the nodes and edges files, return a dict of the name -> relative path"""

    # Note: we are 'prioritizing' the layout of the config over the layout of the files on disk:
    # 1) the `nodes`/`edges` network keys will still have the same number of elements
    #    after writing the new config (unless populations aren't used)
    # 2) The layout of the files may be slightly different; if the config has a single population
    #    in the dict, the output population will be writen to $population_name/$original_filename.h5
    #    if it has multiple elements, it will be written to
    #    $original_parent_dir/$original_filename.h5
    #
    # See tests for more clarity
    node_populations_to_paths, edge_populations_to_paths = {}, {}

    node_populations_to_paths = extract_population_paths(key="nodes", networks=networks)
    edge_populations_to_paths = extract_population_paths(key="edges", networks=networks)

    return node_populations_to_paths, edge_populations_to_paths
