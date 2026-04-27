Command-Line Interface
======================

Commands
--------

Building Synthetic Atlases
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following subcommands can be used with: ``brainbuilder atlases``

* ``column``          Build synthetic hexagonal column atlas
* ``hyperrectangle``  Build synthetic hyper-rectangle atlas

Building CellCollection
~~~~~~~~~~~~~~~~~~~~~~~

The following subcommands can be used with: ``brainbuilder cells``

* ``assign-emodels``   Assign 'me_combo' property
* ``assign-emodels2``  Assign 'me_combo' property; write me_combo.tsv
* ``place``            Generate cell positions and me-types


Tools for working with MVD3
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following subcommands can be used with: ``brainbuilder mvd3``

* ``add-property``    Add property to MVD3 based on volumetric data
* ``merge``           Merge multiple MVD3 files
* ``reorder-mtypes``  Align /library/mtypes with builder recipe


Tools for working with NRN files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``from-sonata``  Convert SONATA file to partial nrn.h5
* ``from-syn2``    Convert SYN2 file to partial nrn.h5
* ``merge``        Merge utility tool for nrn.h5 Blue Brain synapse file format.


Tools for working with SONATA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following subcommands can be used with: ``brainbuilder sonata``

* ``from-mvd3``                            Convert MVD3 to SONATA nodes
* ``from-syn2``                            Convert SYN2 to SONATA edges
* ``network-config``                       Write SONATA network config
* ``update-morphologies``                  Update h5 morphologies to not include single child parents
* ``update-edge-population``               Given h5_updates from removing single child parents
* ``update-edge-pos``                      Using section_id, segment_id and offset, create `SONATA` position
* ``simple-split-subcircuit``              Split a subset of nodes and edges out of node and edges files
* ``split-subcircuit``                     Based on a `circuit_config`; split out a nodeset
* ``node-set-from-targets``                Convert .target files to node_sets
* ``clip-morphologies``                    Copy morphologies referenced by a population to a separate output directory
* ``update-edge-section-types``            Update edge afferent/efferent section types using section_id
* ``update-projection-efferent-edge-type`` Write projections' efferent section types as axons

For the commands starting with _update-_, read more in :ref:`SONATA: Single Child Reindex`

For the commands dealing with circuit splitting, read more in :ref:`SONATA: Split Circuit`


Tools for working with SYN2
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following subcommands can be used with: ``brainbuilder syn2``

* ``check``   check SYN2 invariants
* ``concat``  concatenate multiple SYN2 files


Tools for working with .target files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following subcommands can be used with: ``brainbuilder targets``

* ``from-mvd3``  Generate .target file from MVD3 (and target definition YAML)
* ``node-sets``  Generate JSON node sets from MVD3 (and target definition YAML)


Tools for converting Allen's V1 circuit into SONATA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following subcommands can be used with: ``brainbuilder sonata``

 * ``convert-allen-circuit`` Convert nodes and inner connectivity edges file
 * ``convert-allen-projection-edges`` Convert projection edges file
 * ``precompute-allen-synapse-locations`` Precompute synapse locations
 * ``add-nodes-attributes`` Add threshold_current and holding_current data to the nodes file

For example, for the current V1 circuit:

.. code-block:: bash

    brainbuilder sonata precompute-allen-synapse-locations -o obi_circuit2 --nodes-file network/v1_nodes.h5 --node-types-file network/v1_node_types.csv --morphology-dir components/morphologies --edges-files network/v1_v1_edges.h5 network/v1_v1_edge_types.csv —-edges-files network/lgn_v1_edges.h5 network/lgn_v1_edge_types.csv --edges-files network/bkg_v1_edges.h5 network/bkg_v1_edge_types.csv
    brainbuilder sonata convert-allen-circuit -o 260105 --node-types-file network/v1_node_types.csv --nodes-file network/v1_nodes.h5 --edges-file network/v1_v1_edges.h5 --edge-types-file network/v1_v1_edge_types.csv --precomputed-edges-file obi_circuit2/v1_v1_syn_locations.h5 --syn-parameter-dir components/synaptic_models
    brainbuilder sonata convert-allen-projection-edges --target-nodes-file network/v1_nodes.h5 --target-node-types-file network/v1_node_types.csv --n-source-nodes 17400  --edges-file network/lgn_v1_edges.h5  --edge-types-file network/lgn_v1_edge_types.csv --precomputed-edges-file obi_circuit2/lgn_v1_syn_locations.h5 -o 260105 —-syn-parameter-dir components/synaptic_models
    brainbuilder sonata convert-allen-projection-edges --target-nodes-file network/v1_nodes.h5 --target-node-types-file network/v1_node_types.csv --n-source-nodes 1 --edges-file network/bkg_v1_edges.h5  --edge-types-file network/bkg_v1_edge_types.csv --precomputed-edges-file obi_circuit2/bkg_v1_syn_locations.h5 -o 260105 --syn-parameter-dir components/synaptic_models
    brainbuilder sonata add-nodes-attributes -o output_dir --nodes-file obi_circuit2 nodes_biophysical.h5 --attributes-file network/dynamic_parameters.csv
