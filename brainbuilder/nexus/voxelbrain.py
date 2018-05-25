# pylint: skip-file

from voxcell.nexus.voxelbrain import *

from voxcell.utils import deprecate

deprecate.fail("""
    'nexus.voxelbrain' module has been moved to `voxcell` library.
    Please change your imports accordingly:
        brainbuilder.nexus.voxelbrain -> voxcell.nexus.voxelbrain
""")
