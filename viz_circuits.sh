#!/bin/bash
# Viz all circuits from test_subsubcircuit_externals_merge
# Usage: ./viz_circuits.sh

set -e

BASE="removeme/test_subsubcircuit_externals_m0"
BB="venv/bin/brainbuilder"

rm -rf removeme
venv/bin/tox -e py312 -- tests/unit/test_sonata/test_split_population.py::test_subsubcircuit_externals_merge --basetemp=removeme -vv

$BB sonata visualize tests/unit/data/sonata/split_subcircuit/circuit_config.json --title "Original (A)"

for dir in "$BASE"/*/; do
    name=$(basename "$dir")
    [[ "$name" == *_fixture ]] && continue
    config="$dir/circuit_config.json"
    if [ -f "$config" ]; then
        $BB sonata visualize "$config" --title "$name"
    fi
done
