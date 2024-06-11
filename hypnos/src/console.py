import sys
from rich import console, traceback

out_console = console.Console(color_system="truecolor")
err_console = console.Console(color_system="truecolor", stderr=True, file=sys.stderr)

traceback.install(console=err_console)


def rprint(*objects, **kwargs):
    return out_console.print(*objects, **kwargs)
