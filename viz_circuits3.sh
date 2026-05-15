#!/bin/bash
# Viz circuits from test_external_nodes_file_contains_all_merged_ids
# Usage: ./viz_circuits3.sh

set -e

BASE="removeme/test_external_nodes_file_conta0"
BB=".tox/py312/bin/brainbuilder"
PICS="removeme/viz_circuits_pics"

rm -rf removeme

.tox/py312/bin/pytest tests/unit/test_sonata/test_split_population.py::test_external_nodes_file_contains_all_merged_ids --basetemp=removeme -vv --cache-clear

mkdir -p "$PICS"

$BB sonata visualize tests/unit/data/sonata/split_subcircuit/circuit_config.json --title "Original" -o "$PICS/Original.png"

for dir in "$BASE"/*/; do
    name=$(basename "$dir")
    [[ "$name" == *_fixture ]] && continue
    config="$dir/circuit_config.json"
    if [ -f "$config" ]; then
        $BB sonata visualize "$config" --title "$name" -o "$PICS/${name}.png"
    fi
done

open "$PICS"/*.png

echo "Pictures saved to $PICS/"