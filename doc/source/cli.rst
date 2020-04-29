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

The following subcommands can be used with: ``brainbuilder nrn``

* ``from-sonata``  Convert SONATA file to partial nrn.h5
* ``from-syn2``    Convert SYN2 file to partial nrn.h5
* ``merge``        Merge utility tool for nrn.h5 Blue Brain synapse file format.


Tools for working with SONATA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following subcommands can be used with: ``brainbuilder sonata``

* ``from-mvd3``                   Convert MVD3 to SONATA nodes
* ``from-syn2``                   Convert SYN2 to SONATA edges
* ``network-config``              Write SONATA network config


Genes / proteins assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following subcommands can be used with: ``brainbuilder subcellular``

* ``assign``  Assign subcellular data


Tools for working with SYN2
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following subcommands can be used with: ``brainbuilder syn2``

* ``check``   check syn2 invariants
* ``concat``  concatenate multiple syn2 files


Tools for working with .target files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following subcommands can be used with: ``brainbuilder targets``

* ``from-mvd3``  Generate .target file from MVD3 (and target definition YAML)
* ``node-sets``  Generate JSON node sets from MVD3 (and target definition YAML)
