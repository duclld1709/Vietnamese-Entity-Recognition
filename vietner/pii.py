"""Rule-based recognisers for structured Vietnamese personal data.

The NER model covers names, organisations and locations. Identifiers such as phone
numbers or citizen ID numbers follow fixed formats, so validated patterns beat a
statistical model on precision. Ambiguous digit strings (bank accounts, 9-digit
CMND, tax codes) are only reported when a context keyword precedes them.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vietner.schemas import Entity

# Province/city codes that open a 12-digit CCCD number (Circular 07/2016/TT-BCA).
_PROVINCE_CODES = (
    "001 002 004 006 008 010 011 012 014 015 017 019 020 022 024 025 026 027 030 031 033 034 035 036 037 038 "
    "040 042 044 045 046 048 049 051 052 054 056 058 060 062 064 066 067 068 070 072 074 075 077 079 080 082 "
    "083 084 086 087 089 091 092 093 094 095 096"
)
CCCD_PROVINCES = frozenset(_PROVINCE_CODES.split())
MOBILE_PREFIXES = ("03", "05", "07", "08", "09")
CONTEXT_WINDOW = 40


@dataclass(frozen=True)
class Recognizer:
    label: str
    pattern: re.Pattern
    score: float
    validate: Callable[[str], bool] = lambda _: True
    context: tuple[str, ...] = ()
    context_required: bool = False
    context_score: float = 0.99


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _valid_phone(s: str) -> bool:
    d = _digits(s)
    if d.startswith("84"):
        d = "0" + d[2:]
    return (len(d) == 10 and d[:2] in MOBILE_PREFIXES) or (len(d) == 11 and d.startswith("02"))


def _valid_cccd(s: str) -> bool:
    return s[:3] in CCCD_PROVINCES


def _luhn(s: str) -> bool:
    d = [int(c) for c in _digits(s)][::-1]
    if not 13 <= len(d) <= 19:
        return False
    total = sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])
    return total % 10 == 0


def _valid_date(s: str) -> bool:
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if len(nums) != 3:
        return False
    day, month, year = nums
    return 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100


_PHONE_CONTEXT = ("sđt", "số điện thoại", "điện thoại", "đt", "hotline", "liên hệ", "gọi", "phone", "zalo")
_ID_CONTEXT = ("cccd", "căn cước", "cmnd", "chứng minh", "định danh", "số id", "hộ chiếu")
_NB = r"(?<![\w.])"  # no word char or dot before
_NA = r"(?![\w])"  # no word char after

RECOGNIZERS: tuple[Recognizer, ...] = (
    Recognizer("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), 0.99),
    Recognizer("URL", re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+[^\s<>\"'.,;:!?)]"), 0.95),
    Recognizer(
        "PHONE",
        # mobile: 0 + [35789] + 8 digits; landline: 0 + 2 + 9 digits; "+84"/"84" may replace the 0
        re.compile(_NB + r"(?:\+?84[\s.\-]?|0)(?:[35789](?:[\s.\-]?\d){8}|2(?:[\s.\-]?\d){9})" + _NA),
        0.9,
        _valid_phone,
        context=_PHONE_CONTEXT,
    ),
    Recognizer("CREDIT_CARD", re.compile(_NB + r"\d{4}(?:[\s\-]?\d{4}){2}[\s\-]?\d{1,7}" + _NA), 0.95, _luhn),
    Recognizer("ID_NUMBER", re.compile(_NB + r"0\d{11}" + _NA), 0.85, _valid_cccd, context=_ID_CONTEXT),
    Recognizer("ID_NUMBER", re.compile(_NB + r"\d{9}" + _NA), 0.9, context=_ID_CONTEXT, context_required=True),
    Recognizer(
        "TAX_CODE",
        re.compile(_NB + r"\d{10}(?:-\d{3})?" + _NA),
        0.9,
        context=("mst", "mã số thuế", "mã số doanh nghiệp"),
        context_required=True,
    ),
    Recognizer(
        "BANK_ACCOUNT",
        re.compile(_NB + r"\d(?:[\s.\-]?\d){5,18}" + _NA),
        0.9,
        context=("stk", "số tài khoản", "tài khoản", "tk ngân hàng", "account"),
        context_required=True,
    ),
    Recognizer(
        "LICENSE_PLATE",
        re.compile(r"\b\d{2}[A-Z]{1,2}\d?[\s\-.]?\d{3}\.?\d{2}\b"),
        0.9,
    ),
    Recognizer(
        "DATE",
        re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4}\b|\bngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}\b", re.I),
        0.8,
        _valid_date,
        context=("sinh", "ngày sinh", "dob", "sn"),
    ),
)


def _has_context(text: str, start: int, keywords: Iterable[str]) -> bool:
    window = text[max(0, start - CONTEXT_WINDOW) : start].lower()
    return any(re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", window) for k in keywords)


def find_pii(text: str, recognizers: Iterable[Recognizer] = RECOGNIZERS) -> list[Entity]:
    """Return all rule matches (possibly overlapping; :func:`resolve_overlaps` picks winners)."""

    found: list[Entity] = []
    for r in recognizers:
        for m in r.pattern.finditer(text):
            value = m.group()
            if not r.validate(value):
                continue
            has_ctx = bool(r.context) and _has_context(text, m.start(), r.context)
            if r.context_required and not has_ctx:
                continue
            found.append(
                Entity(
                    label=r.label,
                    text=value,
                    start=m.start(),
                    end=m.end(),
                    score=r.context_score if has_ctx else r.score,
                    source="rule",
                )
            )
    return found


def resolve_overlaps(entities: Iterable[Entity]) -> list[Entity]:
    """Greedy non-overlapping selection: higher score, then longer span, then rules first."""

    ranked = sorted(entities, key=lambda e: (-e.score, -(e.end - e.start), e.source != "rule", e.start))
    chosen: list[Entity] = []
    for e in ranked:
        if all(e.end <= c.start or e.start >= c.end for c in chosen):
            chosen.append(e)
    return sorted(chosen, key=lambda e: e.start)
