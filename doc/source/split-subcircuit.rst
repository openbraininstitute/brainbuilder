SONATA: Split Circuit
=====================

Two methods exist for splitting up SONATA based circuits; both rely on specifying a nodeset.

split-subcircuit
~~~~~~~~~~~~~~~~

This command splits out a subcircuit from a fully compatible SONATA circuit.
It assumes that a circuit_config.json exists, with the BBP extensions, that references all the node and edge populations that may be impacted by a particular nodeset.
By specifying which nodeset is to be extracted, a new circuit can be created.

Two options are worth explaining further:

* `--include-virtual`:
  Circuits often have virtual nodes that innervate the subcircuit that is being extracted.
  If these virtual nodes should also be extracted, this option is used to specify that.
  It creates a population per virtual population, including the nodes file, of only the nodes that project into the circuit that is extracted.
* `--create-external`:
  When splitting out a subcircuit, sometimes it's useful to maintain connectivity that projects onto the nodes of the subcircuit, but whose source nodes aren't part of the node set.
  This is considered `external` connectivity, and can be extracted with this option.
  It creates virtual nodes and edges for all the connectivity that *targets* the extracted nodes.
  However, it does not remove the parameters associated with the edges - that is up to the user if they desire.

.. code-block:: bash

    $ CIRCUIT_CONFIG=... # path to a SONATA `circuit_config.json`
    $ NODESET_NAME=...   # name of the nodeset, as referenced in the `node_sets_file` in the above `circuit_config.json`
    $ OUTPUT=...         # directory to store output

    $ mkdir -p $OUTPUT
    $ brainbuilder                 \
      sonata split-subcircuit      \
      --circuit $CIRCUIT_CONFIG    \
      --nodeset $NODESET_NAME      \
      --output $OUTPUT
    # specify --include-virtual / --create-external as desired


.. note::

    An attempt is made to make a valid circuit_config.json.
    However due to the nature of the paths (like the locations of the morphologies), and the expansion of the original "manifest" keys, it requires human intervention post-split

simple-split-subcircuit
~~~~~~~~~~~~~~~~~~~~~~~

This method is doesn't require a circuit_config, and is simpler and lower level.
It's goal is to split out a single population, defined by a nodeset, from a nodes and edges file.
It is useful if a `circuit_config.json` doens't exist, or if the circuit is simple.

An example of how to run it is:

.. code-block:: bash

    $ NODESET_NAME=...  #
    $ NODESET_PATH=...  #
    $ NODES_PATH=...    # Path to SONATA edges file
    $ EDGES_PATH=...    # Path to SONATA edges file
    $ OUTPUT=...        # directory to store output; will be a new nodes and new edges files

    $ mkdir -p $OUTPUT
    $ brainbuilder                        \
      sonata simple-split-subcircuit      \
      --nodeset $NODESET_NAME             \
      --nodeset-path $NODESET_PATH        \
      --nodes $NODES_PATH                 \
      --edges $EDGES_PATH                 \
      --output $OUTPUT
