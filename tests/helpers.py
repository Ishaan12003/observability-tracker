"""Test helpers.

A plain module rather than ``conftest``: pytest may import a conftest under
more than one name, so importing from it risks two copies of the same object.
"""

from __future__ import annotations

from obspipe.models.document import RawDocument


def text_doc(content: str, name: str = "inline.txt") -> RawDocument:
    """Build a RawDocument straight from a string, for one-off cases."""
    return RawDocument.from_text(content, source_name=name)
