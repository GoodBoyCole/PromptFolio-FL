"""
Modified from https://github.com/facebookresearch/fvcore
"""
__all__ = ["Registry"]


class Registry:
    """A registry providing name -> object mapping, to support
    custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To