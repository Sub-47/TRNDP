_REGISTRY = {}


def register_source(name):
    def decorator(cls):
        _REGISTRY[name.upper()] = cls
        return cls

    return decorator


def get_registered_source(name):
    try:
        return _REGISTRY[name.upper()]
    except KeyError as e:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise ValueError(f"Unknown MAP_SOURCE '{name}'. Registered sources: {available}") from e
