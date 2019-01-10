""" Deprecation utilities. """

import warnings

from brainbuilder.exceptions import BrainBuilderError


class BrainBuilderDeprecationWarning(UserWarning):
    """ brainbuilder deprecation warning. """
    pass


class BrainBuilderDeprecationError(BrainBuilderError):
    """ brainbuilder deprecation error. """
    pass


def fail(msg=None):
    """ Raise a deprecation exception. """
    raise BrainBuilderDeprecationError(msg)


def warn(msg=None):
    """ Issue a deprecation warning. """
    warnings.warn(msg, BrainBuilderDeprecationWarning)
