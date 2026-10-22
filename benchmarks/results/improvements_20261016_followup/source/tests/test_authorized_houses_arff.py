"""Synthetic bytes only. Never opens or downloads any real data."""
import hashlib
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("draft_binary_arff_selector",
    Path(__file__).resolve().parents[1] / "benchmarks/authorized_houses_arff.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def header():
    return (b"% header comment\n@RELATION 'synthetic houses'\n" + b"".join(
        f"@ATTRIBUTE '{name}' NUMERIC\n".encode() for name in selector.ATTRIBUTES) + b"@DATA\n")


def document(rows):
    return header() + b"\n".join(rows) + b"\n"


def select(raw, requested=(0,), rows=3, **kwargs):
    options = {"expected_md5": hashlib.md5(raw).hexdigest(),
               "expected_sha256": hashlib.sha256(raw).hexdigest(), "expected_rows": rows}
    options.update(kwargs)
    return selector.select_authorized_rows(raw, requested, **options)


GOOD = b"1,2,3,4,5,6,7,8,9"


def test_excluded_invalid_utf8_and_nonnumeric_values_are_never_decoded_or_tokenized(monkeypatch):
    raw = document([b"\xff,excluded_secret", GOOD, b"not numeric and no delimiters"])
    parsed = []
    original = selector._parse_selected
    def observe(line, row):
        parsed.append(row)
        return original(line, row)
    monkeypatch.setattr(selector, "_parse_selected", observe)
    values, receipt = select(raw, (1,))
    assert values == [[float(i) for i in range(1, 10)]]
    assert parsed == [1]
    assert receipt["excluded_rows_decoded"] == receipt["excluded_rows_tokenized"] == 0


def test_request_order_is_preserved():
    raw = document([GOOD, b"2,2,2,2,2,2,2,2,2", b"3,3,3,3,3,3,3,3,3"])
    values, receipt = select(raw, (2, 0))
    assert values == [[3.] * 9, [float(i) for i in range(1, 10)]]
    assert receipt["selected_rows"] == 2
    assert receipt["total_rows"] == 3
    assert receipt["attributes"] == list(selector.ATTRIBUTES)


@pytest.mark.parametrize("requested", [(0, 0), (-1,), (3,), (True,), (1.,), [], (), [0]])
def test_invalid_requests_fail(requested):
    with pytest.raises(ValueError, match="invalid_.*"):
        select(document([GOOD] * 3), requested)


@pytest.mark.parametrize("bad,code", [
    (b"\xff,2,3,4,5,6,7,8,9", "invalid_selected_encoding"),
    (b"private_secret,2,3,4,5,6,7,8,9", "invalid_selected_numeric_field"),
    (b"?,2,3,4,5,6,7,8,9", "invalid_selected_numeric_field"),
    (b"nan,2,3,4,5,6,7,8,9", "invalid_selected_numeric_field"),
    (b"Inf,2,3,4,5,6,7,8,9", "invalid_selected_numeric_field"),
    (b"1e999,2,3,4,5,6,7,8,9", "nonfinite_selected_numeric_field"),
    (b"1_000,2,3,4,5,6,7,8,9", "invalid_selected_numeric_field"),
    (b"1,2,3,4,5,6,7,8", "invalid_selected_field_count"),
    (GOOD + b",10", "invalid_selected_field_count"),
    (GOOD + b" % private_secret", "unsupported_instance_syntax"),
])
def test_selected_invalid_fields_fail_without_echoing_content(bad, code):
    with pytest.raises(ValueError) as error:
        select(document([bad, GOOD, GOOD]))
    assert str(error.value) == code + " at row 0"


@pytest.mark.parametrize("bad", [b"{0 1,1 2}", GOOD + b",{2}", b'"multiline',
                                  b"1,2,3\\", b"@data", b"quoted'value", GOOD + b" % comment"])
def test_unsupported_structure_on_excluded_rows_fails_generically(bad):
    with pytest.raises(ValueError, match="^unsupported_instance_syntax at row 0$"):
        select(document([bad, GOOD, GOOD]), (1,))


def test_crlf_blank_comments_and_finite_scientific_numbers():
    row = b" +1., -.5, 2e2, 3E-2, 4, 5, 6, 7, 8 "
    raw = (header() + b"\n%\xff opaque comment\n" + row + b"\n\t\n" + GOOD + b"\n").replace(b"\n", b"\r\n")
    values, _ = select(raw, (0,), rows=2)
    assert values == [[1., -.5, 200., .03, 4., 5., 6., 7., 8.]]


def test_embedded_cr_fails_without_decoding():
    with pytest.raises(ValueError, match="^unsupported_line_ending$"):
        select(document([b"secret\r\xff", GOOD, GOOD]), (1,))


@pytest.mark.parametrize("mutation", ["names", "type", "duplicate", "extra", "missing", "relation", "encoding", "data"])
def test_schema_guards(mutation):
    raw = document([GOOD] * 3)
    if mutation == "names":
        raw = raw.replace(b"'MedianIncome'", b"'WrongName'")
    elif mutation == "type":
        raw = raw.replace(b"NUMERIC", b"REAL", 1)
    elif mutation == "duplicate":
        raw = raw.replace(b"'HousingMedianAge'", b"'MedianIncome'")
    elif mutation == "extra":
        raw = raw.replace(b"@DATA", b"@ATTRIBUTE 'extra' NUMERIC\n@DATA")
    elif mutation == "missing":
        raw = raw.replace(b"@ATTRIBUTE 'MedianIncome' NUMERIC\n", b"")
    elif mutation == "relation":
        raw = raw.replace(b"@DATA", b"@RELATION second\n@DATA")
    elif mutation == "encoding":
        raw = raw.replace(b"synthetic houses", b"\xffsecret")
    else:
        raw = raw[:raw.index(b"@DATA")]
    with pytest.raises(ValueError, match="invalid_header|unsupported_header|missing_data_section"):
        select(raw)


@pytest.mark.parametrize("count", [2, 4])
def test_total_row_count_guard(count):
    with pytest.raises(ValueError, match="^row_count_mismatch$"):
        select(document([GOOD] * count))


@pytest.mark.parametrize("options", [{"expected_md5": "0" * 32}, {"expected_sha256": "0" * 64},
                                    {"expected_sha256": None}, {"expected_md5": "X" * 32}])
def test_hash_guards_run_before_any_header_or_data_decode(monkeypatch, options):
    def forbidden(*args):
        raise AssertionError("hash rejection must precede decoding")
    monkeypatch.setattr(selector, "_decode_header", forbidden)
    monkeypatch.setattr(selector, "_parse_selected", forbidden)
    with pytest.raises(ValueError, match="raw_hash_mismatch|invalid_expected_hashes"):
        select(document([GOOD] * 3), **options)


def test_parser_requires_caller_source_sha_and_has_no_acquisition_entry_point():
    assert selector.DRAFT_ONLY is False
    assert selector.HOUSES_SHA256 is None
    assert selector.HOUSES_MD5 == "5edd572964b0653b789bfc5e5c0293a8"
    assert selector.HOUSES_ROWS == 20640
    assert not hasattr(selector, "main")
