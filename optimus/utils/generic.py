"""Utilities for generic frequently used functions."""


class AnsiEscapeFormat:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
    ITALIC = "\033[3m"


def bold_ul_text(string):
    return (
            AnsiEscapeFormat.BOLD
            + AnsiEscapeFormat.UNDERLINE
            + string
            + AnsiEscapeFormat.END
    )


def bold_ul_red_text(string):
    return (
            AnsiEscapeFormat.BOLD
            + AnsiEscapeFormat.UNDERLINE
            + AnsiEscapeFormat.RED
            + string
            + AnsiEscapeFormat.END
    )


def chunker(seq, size):
    """
    To split a sequence (list, tuples,...) by a specific size.
    Imported from https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
    """
    return (seq[pos: pos + size] for pos in range(0, len(seq), size))
