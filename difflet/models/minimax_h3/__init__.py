"""MiniMax-H3 model integration, with tensor contracts loaded only when requested."""

__all__ = ["MiniMaxH3T2VALayout", "build_padded_t2va_layout", "build_t2va_layout"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from difflet.models.minimax_h3 import contracts

    value = getattr(contracts, name)
    globals()[name] = value
    return value
