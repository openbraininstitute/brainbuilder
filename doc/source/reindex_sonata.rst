SONATA: Single Child Reindex
============================
Brainbuilder must be installed with `reindex` or `all` extras.

.. code-block:: bash

    pip install brainbuilder[reindex]

The Problem
-----------

The `ASCII` and `H5` file formats allow for morphologies with single children parents, however, `SWC` files don't allow that, so circuits aiming for proper `SONATA` support must be 'reindexed'.
Thus the following transform must happen:

.. image:: _ static/reindex_morphology.svg

In this cartoon, the left morphology has 3 sections (A, B, C), each composed of two segments (and multi-children at the end of C).
They are, however, single children, and thus must be merged together.
This result of the merging process is on the right, and it displays the 'reindex' portion of the problem.
The segments need to be updated, first in the morphology, but also in the connectivity files.


This is a three step process:

#. Detect, record and rewrite the morphologies to have no single children
#. Use the record from above to update the section/segment information of the connectivity
#. For SONATA compatibility, update the `*_section_pos` reserved edge attributes_.

Usage
-----

0. Copy the original files:

.. code-block:: bash

    # make useful directories
    $ mkdir morphologies sonata

    # copy morphologies
    $ cp -a original/path/to/h5v1/ morphologies/h5-orig

    # copy SONATA nodes
    $ cp -a original/path/to/nodes.h5 sonata/nodes.h5

    # copy SONATA edges
    $ cp -a original/path/to/edges.h5 sonata/edges.h5


.. note::

    Edge and population files are updated in place, so always best to make a copy


1. Update the morphologies to not have single children, recording the changes:

.. code-block:: bash

    $ brainbuilder sonata update-morphologies \
        --h5-morphs morphologies/h5-orig      \
        -o morphologies/h5


2. Update section/segments of sonata edge files:

.. code-block:: bash

    $ brainbuilder sonata update-edge-population       \
        --h5-updates morphologies/h5/h5_updates.json   \
        --nodes sonata/nodes.h5 sonata/edges.h5


3. update the `*_section_pos` reserved edge attributes:

.. code-block:: bash

    $ brainbuilder sonata update-edge-pos \
        --morph-path morphologies/h5      \
        --nodes sonata/nodes.h5           \
        sonata/edges.h5


.. _attributes: https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#edges---optional-reserved-attributes
