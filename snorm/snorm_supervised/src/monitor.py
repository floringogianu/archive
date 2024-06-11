""" Ză monităr.
"""

from collections import OrderedDict


def append_to_leafs(values, traces):
    """ Here we get a nested dictionary of values, and we add them to the trace.
    """
    assert isinstance(values, dict)
    stack = [(values, traces)]
    while stack:
        vals, trace = stack.pop()
        for key, value in vals.items():
            if isinstance(value, dict):
                stack.append((value, traces.setdefault(key, OrderedDict())))
            else:
                trace.setdefault(key, []).append(value)


class Monitor:
    """ The simplest monitor. I have implemented more intricated versions of this, but
        as I got older I realised that life is made out of simple, beautiful things.
    """

    def __init__(self, *callbacks, printer=None):
        self._callbacks = OrderedDict(callbacks)
        self._trace = {}
        self._printer = printer

    def add_callback(self, name, callback):
        """ We add some callback.
        """
        self._callbacks[name] = callback

    def __call__(self, *args, values=None, **kwargs):
        if values is not None:
            append_to_leafs(values, self._trace)
        for name, callback in self._callbacks.items():
            result = callback(*args, **kwargs)
            if result is not None:
                append_to_leafs(result, self._trace.setdefault(name, OrderedDict()))
        if self._printer is not None:
            self._printer(self)

    @property
    def trace(self):
        """ The trace with all the series.
        """
        return self._trace

    def last(self, keys, last_n=1):
        """ Returns the last n values.
        """

        keys = keys.split("/") if isinstance(keys, str) else list(keys)
        tree = self._trace
        for key in keys:
            tree = tree[key]
        return tree[-1] if last_n == 1 else tree[-last_n:]
