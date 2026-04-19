"""Input ingestion and local parsing helpers for RDDL and RDDL+."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rddl2puffer.rddl_plus.parser import RDDLReader, parse_rddl_text


@dataclass(frozen=True, slots=True)
class ParsedRDDLSource:
    """Raw RDDL source bundle awaiting real parsing and grounding."""

    domain_text: str
    instance_text: str
    nonfluents_text: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedRDDLProgram:
    """Parsed local AST plus the normalized source bundle used to build it."""

    source: ParsedRDDLSource
    ast: object


def ingest_rddl_sources(
    domain_text: str,
    instance_text: str,
    nonfluents_text: str | None = None,
) -> ParsedRDDLSource:
    """Store raw RDDL sources until a real parser is wired in.

    The scaffold intentionally keeps ingestion separate from grounding so the
    schema and IR layers can evolve independently.
    """

    return ParsedRDDLSource(
        domain_text=domain_text,
        instance_text=instance_text,
        nonfluents_text=nonfluents_text,
    )


def parse_rddl_sources(
    domain_text: str,
    instance_text: str,
    nonfluents_text: str | None = None,
) -> ParsedRDDLProgram:
    """Parse in-memory RDDL source text through the local parser fork."""

    source = ingest_rddl_sources(
        domain_text=domain_text,
        instance_text=instance_text,
        nonfluents_text=nonfluents_text,
    )
    combined = _combine_rddl_sources(source)
    return ParsedRDDLProgram(source=source, ast=parse_rddl_text(combined))


def parse_rddl_files(
    domain_path: str | Path,
    instance_path: str | Path,
    nonfluents_path: str | Path | None = None,
) -> ParsedRDDLProgram:
    """Parse a domain/instance pair from disk through the local parser fork."""

    if nonfluents_path is not None:
        reader = RDDLReader(str(domain_path), str(instance_path), str(nonfluents_path))
    else:
        reader = RDDLReader(str(domain_path), str(instance_path))
    domain_text = Path(domain_path).read_text(encoding="utf-8")
    instance_text = Path(instance_path).read_text(encoding="utf-8")
    nonfluents_text = (
        Path(nonfluents_path).read_text(encoding="utf-8") if nonfluents_path is not None else None
    )
    source = ingest_rddl_sources(
        domain_text=domain_text,
        instance_text=instance_text,
        nonfluents_text=nonfluents_text,
    )
    return ParsedRDDLProgram(source=source, ast=parse_rddl_text(reader.rddltxt))


def _combine_rddl_sources(source: ParsedRDDLSource) -> str:
    parts = [source.domain_text.rstrip(), source.instance_text.rstrip()]
    if source.nonfluents_text:
        parts.insert(1, source.nonfluents_text.rstrip())
    return "\n\n".join(part for part in parts if part) + "\n"
