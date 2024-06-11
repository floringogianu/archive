""" I am gonna regret this, right?"""
from rich.console import Console

__all__ = ["console", "cstr"]

console = Console()


def cstr(x, *args, **kwargs):
    """ Utility function for capturing rich.console formatting.
        A bit of a hack used with logging/rlog.
    """
    with console.capture() as capture:
        console.print(x, *args, **kwargs)
    return capture.get()
