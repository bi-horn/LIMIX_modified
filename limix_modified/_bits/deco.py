#Handling None Inputs: If you have functions that may receive None as an argument and you want them to gracefully handle this case by returning None instead of raising an error, then applying this decorator can simplify the implementation.


__all__ = ["return_none_if_none"]


def return_none_if_none(f):
    """ Return ``None`` if first argument is ``None``. """
    from functools import wraps

    @wraps(f)
    def new_func(x, *args, **kwargs):
        if x is None:
            return None
        return f(x, *args, **kwargs)

    return new_func
