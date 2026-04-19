"""Local parser fork for RDDL and future RDDL+ extensions."""

from __future__ import annotations

from rddl2puffer.rddl_plus.parser.parser import RDDLParser
from rddl2puffer.rddl_plus.parser.reader import RDDLReader


def parse_rddl_text(source: str):
    """Parse combined RDDL source text into the local AST."""

    parser = RDDLParser()
    parser.build(write_tables=False, debug=False)
    return parser.parse(source)


__all__ = ["RDDLParser", "RDDLReader", "parse_rddl_text"]
