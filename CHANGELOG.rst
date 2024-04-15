Changelog
=========

## 0.19.1
  * Make ``update_edge_pos`` parallel, require ``--direction``
  * Apply black and isort
  * Add ``--log-level`` option to the main CLI.
  * Include app in coverage calculation.
  * Add tests for app/sonata.
  * when converting .target files to node_sets, warn that `population` should be
    added when `node_id` are created.

## 0.19.0
  * Enforce layer to be a string (NSETM-2261)

## 0.18.7
  * Fix functional tests with snap 2.0
  * Silence nptyping DeprecationWarnings

## 0.18.6
  * Make circuit splitting compatible with snap 2.0

## 0.18.5
  * Remove subcellular querier (BRBLD-97)

## 0.18.4
  * Zero close to zero values (NSETM-2128)
  * Add curation facilities to update dtypes of a circuit (NSETM-2124)
  * Fix implicit dependencies and tests with snap 1.0 (NSETM-2123)
  * Use pytest-basetemp-permissions plugin (NSETM-2125)
  * Use pytest tmp_path fixture for tests, add ruff
  * Update for python 3.10
  * teach `target node_sets --full-hierarchy`, which includes, from leaf to root, all the region names as node_sets

## 0.18.3
  * Use bool instead of np.bool.

## 0.18.2
  * Relax zero cell count assertion to a warning in `place`.
  * Remove `assign_emodels2`; was a work around for the NCX project
  * be more relaxed about matching `layer` and `subregion`
  * for emodel assignment, try `layer` first, and fall back to `subregion` columns

## 0.18.1
  * Densities are now loaded using float64 precision. Higher precision aids in calculating
    correctly the total counts when accumulation of small and large densities is involved.
  * `layer` made an optional trait in `brainbuilder.app.cells.place`
  * `subregion` added as cells property in `brainbuilder.app.cells.place`

## 0.18.0
  * Add simple circuit splitting for SONATA (BRBLD-90).
  * Add subcircuit splitting for SONATA (BRBLD-90).
  * teach `node-set-from-targets` to take paths to files, add to documentation
  * Add support for copying the used morphologies from a SONATA circuit (BRBLD-93).

## 0.17.0
  * Update ``write_network_config`` to be compatible with the new BBP sonata specs, moved from
    ``brainbuilder.utils.sonata.convert`` to ``brainbuilder.utils.sonata.config`` (NSETM-1526).
  * Deprecate `FAST-HEMISPHERE` for assigning the `hemisphere` cells property,
    and support loading it from a volumetric dataset (BRBLD-89).

## 0.16.2
  * Add more intelligent target to node set converter (BBPP82-514)
  * Use pytest for tests (NSETM-1543)
  * Deprecate ``brainbuilder.utils.sonata.convert.write_network_config``. Use circuit-build
    project instead (NSETM-1526).
  * Fix ``brainbuilder.utils.sonata.convert.write_node_set_from_targets`` due to optimization
    of targets in bluepy==2.4.1.

## 0.16.1
  * Fix compat for bluepy>=2.3

## 0.16.0
  * Refactor neurondb functions from `brainbuilder.utils.bbp`. The previous API must be changed as:

    - ``load_neurondb(file, False)`` => ``load_neurondb(file)``
    - ``load_neurondb(file, True)`` => ``load_extneurondb(file)``
    - ``load_neurondb_v3(file)`` => ``load_extneurondb(file)``

## 0.15.1
  * Introduce [reindex] extras
  * Move morph-tool to test dependencies

## 0.15.0
  * Drop python 2 support
  * Add 'split_population' SONATA command to break monolithic node/edge files into smaller version
  * BBPP82-499: Add `cells positions_and_orientations` command to create a sonata file to be used by the web Cell Atlas
    and by Point neuron whole mouse brain
  * Allow users to set the seed of the numpy random generator when calling `cell_positions.create_cell_positions`
  * new python API `brainbuilder.utils.bbp.load_cell_composition` to load cell composition yaml
  * Check if all the mecombo from a cell file are present in the emodel release
    in `utils.sonata.convert._add_me_info` function. If not raise.

## 0.14.1
  * Fix bugs du to voxcell >= 3.0.0
  * Add a test to validate the behaviour of `utils.sonata.convert.write_network_config`

## 0.14.0
  * Add 'functional' tox env for functional tests
  * Add a new package `utils.sonata`. Move all SONATA related utils there
  * Add a new module `utils.sonata.curate` to curate/fix existing SONATA circuits
  * Fix updating edge positions during reindex of `utils.sonata.reindex`
  * Allow None `mecombo_info_path` in `utils.sonata.convert.provide_me_info`

## 0.13.2
  * Catch duplicate me-combos

## 0.13.1
  * Add "--input" option to "cells.place"
  * Fix losing of SONATA population name at "cells.assign_emodels"

## 0.13.0
  * Create "sonata.provide_me_info". This action provisions SONATA nodes with ME info
  * "sonata.from_mvd3" reuses "sonata.provide_me_info" under the hood

## 0.12.1
  * Fixes to sonata functions
  * "assign_emodels2" function now adds the missing biophysical field
  * "sonata.from_mvd3" remove the library argument (handled by voxcell now)
  * "sonata.from_mvd3" add a model type

## 0.12.0

  * Allows cli with mvd3 inputs/outputs to use sonata files instead. The format detection is done
    using the file extension : '.mvd3' will save/read a 'mvd3' file. For any other file extension,
    sonata is used.
  * "place" cli can output seamlessly sonata or mvd3 files
  * "assign_emodels/assign_emodels2" can use sonata or mvd3 files as input
  * "assign" cli can use sonata or mvd3 files as input
  * rename of "target.from_mvd3" to "target.from_input" and can use both formats as input
  * "target.node_sets" can use both formats as input

## 0.11.10

 * Add atlas based node set with sonata files [NSETM-1010]
 * Change the node_set location inside the sonata config file. Now attached to the circuit not
   the node files

## 0.11.9

 * added reindex for only children, need to convert connectivity to swc
 * updated & fixed documentation
 * Fix empty query_based crash [NSETM-1003]

## 0.11.8

 * atlases creation cli

## 0.11.7

 * Use NodePopulation.from_cell_collection
 * BBPBGLIB-557: use SONATA naming, not syn2
 * Add target to node_set direct converter

## 0.11.6

 * add sonata2nrn converter, so we can build spatial indices

## 0.11.5

 * add syn2 concat and check support
 * BBPP82-94: Add @library enums to mvd3 -> sonata node converter
 * remove seed handling: NSETM-215
