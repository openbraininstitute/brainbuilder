Changelog
=========

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
