"""Select pre-authorized numeric ARFF rows from hash-verified bytes.

No filesystem, network, dataset preparation or fitting entry point is provided.
The caller must freeze authorized global row IDs and both raw-file hashes before
calling this parser. Excluded instance bytes are never decoded or tokenized.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
import struct


DRAFT_ONLY = False
HOUSES_FILE_ID = 22125245
HOUSES_MD5 = "5edd572964b0653b789bfc5e5c0293a8"
HOUSES_SHA256 = None  # Must be frozen after byte-only acquisition, before parsing.
HOUSES_ROWS = 20640
ATTRIBUTES = (
    "MedianIncome", "HousingMedianAge", "TotalRooms", "TotalBedrooms",
    "Population", "Households", "Latitude", "Longitude", "LnMedianHouseValue",
)
_NAME = r"(?:'([^'\\\r\n]+)'|\"([^\"\\\r\n]+)\"|([A-Za-z_][A-Za-z0-9_.-]*))"
_RELATION = re.compile(r"@relation\s+" + _NAME + r"\s*", re.IGNORECASE)
_ATTRIBUTE = re.compile(r"@attribute\s+" + _NAME + r"\s+(?:numeric|real)\s*", re.IGNORECASE)
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_ASCII_SPACE = b" \t\v\f"


def _fail(code, row=None):
    # Never include any source line, token, decoding error or numeric value.
    suffix = "" if row is None else f" at row {row}"
    raise ValueError(code + suffix) from None


def _physical_line(raw_line):
    if raw_line.endswith(b"\n"):
        raw_line = raw_line[:-1]
        if raw_line.endswith(b"\r"):
            raw_line = raw_line[:-1]
    if b"\r" in raw_line:
        _fail("unsupported_line_ending")
    return raw_line.strip(_ASCII_SPACE)


def _decode_header(line):
    try:
        return line.decode("utf-8")
    except UnicodeDecodeError:
        _fail("invalid_header_encoding")


def _parse_selected(line, row):
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        _fail("invalid_selected_encoding", row)
    tokens = text.split(",")
    if len(tokens) != len(ATTRIBUTES):
        _fail("invalid_selected_field_count", row)
    result = []
    for token in tokens:
        token = token.strip(" \t\v\f")
        if _NUMBER.fullmatch(token) is None:
            _fail("invalid_selected_numeric_field", row)
        try:
            value = float(token)
        except (ValueError, OverflowError):
            _fail("invalid_selected_numeric_field", row)
        if not math.isfinite(value):
            _fail("nonfinite_selected_numeric_field", row)
        result.append(value)
    return result


def select_authorized_rows(raw, authorized_rows, *, expected_md5, expected_sha256,
                           expected_rows=HOUSES_ROWS):
    """Return numeric rows in the predeclared request order and a provenance receipt.

    The narrow format is UTF-8 header, NUMERIC/REAL attributes in ATTRIBUTES order,
    unquoted dense comma-separated instances, and LF/CRLF physical lines. Blank
    lines and full-line percent comments are ignored. Selected fields must be
    finite decimal/scientific numbers; missing values are unsupported. Excluded
    data fields are not inspected: only byte-level structural syntax is guarded.
    """
    if (type(raw) is not bytes or type(authorized_rows) is not tuple
            or not authorized_rows or type(expected_rows) is not int or expected_rows <= 0):
        _fail("invalid_request")
    if (any(type(row) is not int or row < 0 or row >= expected_rows for row in authorized_rows)
            or len(set(authorized_rows)) != len(authorized_rows)):
        _fail("invalid_authorized_row_ids")
    if (type(expected_md5) is not str or re.fullmatch(r"[0-9a-f]{32}", expected_md5) is None
            or type(expected_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None):
        _fail("invalid_expected_hashes")
    request_hash = hashlib.sha256(b"".join(struct.pack("<q", row) for row in authorized_rows)).hexdigest()
    md5 = hashlib.md5(raw).hexdigest()
    sha256 = hashlib.sha256(raw).hexdigest()
    if md5 != expected_md5 or sha256 != expected_sha256:
        _fail("raw_hash_mismatch")

    positions = {row: position for position, row in enumerate(authorized_rows)}
    selected = [None] * len(authorized_rows)
    relation_seen, in_data, attributes, row_count = False, False, [], 0
    for raw_line in io.BytesIO(raw):
        line = _physical_line(raw_line)
        if not line or line.startswith(b"%"):
            continue
        if not in_data:
            text = _decode_header(line)
            if text.lower() == "@data":
                if not relation_seen or tuple(attributes) != ATTRIBUTES:
                    _fail("invalid_header_schema")
                in_data = True
                continue
            relation = _RELATION.fullmatch(text)
            attribute = _ATTRIBUTE.fullmatch(text)
            if relation and not relation_seen and not attributes:
                relation_seen = True
            elif attribute and relation_seen and len(attributes) < len(ATTRIBUTES):
                attributes.append(next(value for value in attribute.groups() if value is not None))
            else:
                _fail("unsupported_header")
            continue
        row = row_count
        row_count += 1
        if row_count > expected_rows:
            _fail("row_count_mismatch")
        # Structural byte checks only, including on excluded rows. Do not split,
        # decode, numeric-validate, or print their contents.
        if any(marker in line for marker in (b"{", b"}", b"'", b'"', b"\\", b"@", b"%")):
            _fail("unsupported_instance_syntax", row)
        if row in positions:
            selected[positions[row]] = _parse_selected(line, row)
    if not in_data:
        _fail("missing_data_section")
    if row_count != expected_rows or any(value is None for value in selected):
        _fail("row_count_mismatch")
    return selected, {
        "draft_only": DRAFT_ONLY, "raw_md5": md5, "raw_sha256": sha256,
        "total_rows": row_count, "selected_rows": len(authorized_rows),
        "authorized_rows_sha256": request_hash,
        "authorized_rows_hash_encoding": "ordered little-endian signed int64, no length prefix",
        "attributes": list(ATTRIBUTES), "excluded_rows_decoded": 0,
        "excluded_rows_tokenized": 0, "missing_values_supported": False,
    }
