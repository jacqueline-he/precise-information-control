from rich.console import Console


def fancy_print(console, msg):
    console.print()
    console.print(msg, markup=False, style="plum4")
    console.print()
