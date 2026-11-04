"""Deterministic 50-document demo corpus with independent ground truth.

Each document is rendered from a template whose parties, dates and amounts the
generator chose itself, so the golden record is written from what was *put
into* the document - never from what the pipeline got out of it. That is the
difference between ground truth and a snapshot.

37 documents are clean. 13 are built to fail, each in a different way, and the
sidecar says how. Some failures live in the document (a 20k-char contract, OCR
damage, an honestly ambiguous memo); others are model misbehaviour the fake
backend reproduces on that one document (a hallucinated party, a timeout).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

SEED = 20251102
CORPUS_SIZE = 50

COMPANIES = (
    "Northwind Traders LLC", "Contoso Analytics Inc", "Fabrikam Robotics GmbH",
    "Tailspin Toys Ltd", "Litware Systems Corp", "Adatum Health LLC", "Proseware Labs Inc",
    "Wingtip Logistics Ltd", "Coho Vineyard LLC", "Margie Travel Inc", "Alpine Ski House Ltd",
    "Humongous Insurance Co", "Lucerne Publishing Ltd", "Woodgrove Capital Inc",
    "Blue Yonder Airlines Inc", "Trey Research LLC", "Wide World Importers Ltd",
    "Fourth Coffee Co", "Relecloud Inc", "VanArsdel Ltd", "Graphic Design Institute",
    "Southridge Video Corp", "Datum Freight LLC", "Bellows Media Group",
)
PEOPLE = (
    "Priya Raman", "Daniel Okafor", "Mei Lin", "Lucas Moreau", "Sara Haddad",
    "Tomas Herrera", "Aisha Bello", "Kenji Watanabe", "Elena Petrova", "Marcus Reid",
    "Fatima Zahra", "Oliver Brandt", "Nadia Kowalski", "Rafael Souza", "Ingrid Larsen",
)
STATES = ("the State of Delaware", "the State of New York", "England and Wales", "the State of California")
CONTRACT_KINDS = (
    ("Master Services", "services"), ("Software License", "license"),
    ("Consulting", "consulting"), ("Equipment Lease", "lease"), ("Data Processing", "data"),
)
SERVICES = (
    "Platform engineering", "Data pipeline maintenance", "Security assessment",
    "UX research sessions", "Cloud migration support", "Model evaluation", "On-site training",
    "Network monitoring", "Content localization", "Quality assurance testing",
)

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")


def _long_date(d: date) -> str:
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def _day_month(d: date) -> str:
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def _usd(value: Decimal) -> str:
    return f"${value:,.2f}"


def _key(value: Decimal, currency: str = "USD") -> str:
    return f"{value.quantize(Decimal('0.01'))} {currency}"


def _leader(label: str, amount: str, width: int = 44) -> str:
    """'Subtotal .......... $1,900.00' - long enough that a line is its own context."""
    dots = max(4, width - len(label) - len(amount))
    return f"{label} {'.' * dots} {amount}"


@dataclass
class Doc:
    doc_id: str
    filename: str
    content: str
    expected_type: str
    golden: dict[str, Any]
    expected_failure: str | None = None
    simulated_faults: list[str] = field(default_factory=list)
    expected_warning_codes: list[str] = field(default_factory=list)
    notes: str = ""


# -- templates ------------------------------------------------------------------


def _contract(rng: random.Random, number: int, *, long_tail: bool = False) -> Doc:
    client, contractor = rng.sample(COMPANIES, 2)
    title, slug = rng.choice(CONTRACT_KINDS)
    effective = date(2025, rng.randint(1, 9), rng.randint(1, 28))
    signed = effective - timedelta(days=rng.randint(3, 20))
    fee = Decimal(rng.randrange(18, 140) * 250)
    months = rng.choice((12, 18, 24, 36))
    law = rng.choice(STATES)

    head = [
        f"# {title.upper()} AGREEMENT",
        "",
        f"Agreement No: {slug.upper()}-{4000 + number}",
        "",
        f'This {title} Agreement (the "Agreement") is made between {client} (the "Client") '
        f'and {contractor} (the "Contractor").',
        "",
        "## 1. Scope",
        f"The Contractor shall provide {rng.choice(SERVICES).lower()} and related deliverables "
        "to the Client as described in each statement of work. Each party shall perform its "
        "obligations in good faith.",
        "",
        "## 2. Confidentiality",
        "Each party shall keep confidential all non-public information disclosed by the other "
        "party and shall use it only to perform this Agreement.",
        "",
    ]
    filler: list[str] = []
    if long_tail:
        clauses = (
            "Service levels", "Change control", "Personnel", "Subcontracting", "Audit rights",
            "Insurance", "Intellectual property", "Warranties", "Limitation of liability",
            "Indemnification", "Data protection", "Business continuity", "Records retention",
            "Export control", "Anti-bribery", "Publicity", "Assignment", "Notices",
            "Dispute escalation", "Force majeure", "Severability", "Waiver",
        )
        for index, clause in enumerate(clauses * 2, start=3):
            filler += [
                f"## {index}. {clause}",
                f"The provisions on {clause.lower()} apply to every statement of work issued under "
                "this Agreement. The Contractor shall maintain written procedures covering this "
                "subject, make them available to the Client on reasonable request, and notify the "
                "Client promptly of any material change. Where the Client reasonably objects to a "
                "change, the parties shall meet within ten business days to agree a remedy, and "
                "until a remedy is agreed the previous procedure continues to apply in full.",
                "",
            ]
    n = 3 + len(filler) // 3
    tail = [
        f"## {n}. Term",
        f"The term of this Agreement is {months} months, commencing on the Effective Date of "
        f"{_long_date(effective)}.",
        "",
        f"## {n + 1}. Fees",
        f"The Client shall pay the Contractor a fixed monthly fee of {_usd(fee)} within thirty "
        "days of each statement.",
        "",
        f"## {n + 2}. Termination",
        "Either party may terminate this Agreement for material breach on thirty days written "
        "notice.",
        "",
        f"## {n + 3}. Governing Law",
        f"This Agreement shall be governed by the laws of {law}.",
        "",
        f"IN WITNESS WHEREOF the parties have executed this Agreement on {_long_date(signed)}.",
        "",
        f"Client: {client}",
        f"Contractor: {contractor}",
    ]
    content = "\n".join(head + filler + tail) + "\n"
    doc_id = f"{number:03d}_contract_{slug}"
    return Doc(
        doc_id=doc_id,
        filename=f"{doc_id}.md",
        content=content,
        expected_type="contract",
        golden={
            "intake": {"truncated": False, "high_noise": False},
            "extraction": {
                "parties": sorted([client, contractor], key=str.casefold),
                "dates": sorted([effective.isoformat(), signed.isoformat()]),
                "amounts": [_key(fee)],
            },
            "classification": {"doc_type": "contract"},
            "summarization": {
                "doc_type": "contract",
                "effective_date": effective.isoformat(),
                "parties": sorted([client, contractor], key=str.casefold),
            },
        },
    )


def _invoice(
    rng: random.Random,
    number: int,
    *,
    numeric_dates: bool = False,
    second_currency: bool = False,
) -> Doc:
    vendor, customer = rng.sample(COMPANIES, 2)
    issued = date(2025, rng.randint(1, 10), rng.randint(1, 12 if numeric_dates else 28))
    due = issued + timedelta(days=30)
    if numeric_dates:
        # Day-first and ambiguous on purpose: 03/04 can be read either way.
        while due.day > 12 or due.day == due.month:
            due += timedelta(days=1)
        while issued.day == issued.month:
            issued += timedelta(days=1)
    rate = Decimal(rng.choice((95, 120, 150, 180, 210)))
    items = []
    for service in rng.sample(SERVICES, rng.randint(2, 4)):
        hours = rng.randint(4, 40)
        items.append((f"{service} ({hours} hours)", rate * hours))
    subtotal = sum((v for _, v in items), Decimal("0"))
    tax = (subtotal * Decimal("0.08")).quantize(Decimal("0.01"))
    total = subtotal + tax
    fmt = (lambda d: d.strftime("%d/%m/%Y")) if numeric_dates else _long_date

    lines = [
        "INVOICE",
        "",
        f"Invoice No: INV-{70000 + number * 37}",
        f"Invoice Date: {fmt(issued)}",
        f"Due Date: {fmt(due)}",
        "",
        f"From: {vendor}",
        f"Bill To: {customer}",
        "",
        "Services rendered",
    ]
    lines += [_leader(label, _usd(value)) for label, value in items]
    amounts = [_key(v) for _, v in items]
    eur_key = None
    if second_currency:
        eur = Decimal(rng.randrange(8, 40) * 100)
        lines.append(_leader("Travel expenses (2 days, billed in EUR)", f"EUR {eur:,.2f}"))
        eur_key = _key(eur, "EUR")
    lines += [
        "",
        _leader("Subtotal", _usd(subtotal)),
        _leader("Sales tax at 8 percent", _usd(tax)),
        _leader("Total due", _usd(total)),
        "",
        "Remit within thirty days by bank transfer.",
    ]
    amounts += [_key(subtotal), _key(tax), _key(total)]
    if eur_key:
        amounts.append(eur_key)
    doc_id = f"{number:03d}_invoice_{vendor.split()[0].lower()}"
    return Doc(
        doc_id=doc_id,
        filename=f"{doc_id}.txt",
        content="\n".join(lines) + "\n",
        expected_type="invoice",
        golden={
            "intake": {"truncated": False, "high_noise": False},
            "extraction": {
                "parties": sorted([vendor, customer], key=str.casefold),
                "dates": sorted([issued.isoformat(), due.isoformat()]),
                "amounts": sorted(set(amounts)),
                "total": _key(total),
            },
            "classification": {"doc_type": "invoice"},
            "summarization": {"doc_type": "invoice", "total_stated": _key(total)},
        },
    )


def _report(rng: random.Random, number: int) -> Doc:
    company = rng.choice(COMPANIES)
    author = rng.choice(PEOPLE)
    quarter = rng.randint(1, 3)
    issued = date(2025, quarter * 3 + 1, rng.randint(2, 28))
    spend = Decimal(rng.randrange(40, 400) * 1000)
    growth = rng.randint(3, 19)
    uptime = rng.choice(("99.2", "99.5", "99.9"))
    lines = [
        f"# Q{quarter} 2025 Operations Report - {company}",
        "",
        f"Report date: {_day_month(issued)}",
        f"Prepared by {author}",
        "",
        "## Executive Summary",
        f"This report reviews operating performance for the quarter. The analysis shows order "
        f"volume grew {growth} percent while platform availability held at {uptime} percent "
        "against the SLA.",
        "",
        "## Methodology",
        "Findings are drawn from the operations data warehouse and from milestone reviews held "
        "with each delivery team.",
        "",
        "## Findings",
        f"- Infrastructure spend for the quarter was {_usd(spend)}.",
        "- Two of five delivery milestones slipped by more than two weeks.",
        "- Support backlog fell for the third consecutive quarter.",
        "",
        "## Recommendations",
        "We recommend consolidating monitoring tools and re-baselining the milestone plan.",
    ]
    doc_id = f"{number:03d}_report_{company.split()[0].lower()}"
    return Doc(
        doc_id=doc_id,
        filename=f"{doc_id}.md",
        content="\n".join(lines) + "\n",
        expected_type="report",
        golden={
            "intake": {"truncated": False, "high_noise": False},
            "extraction": {"parties": [], "dates": [issued.isoformat()], "amounts": [_key(spend)]},
            "classification": {"doc_type": "report"},
            "summarization": {"doc_type": "report"},
        },
    )


def _letter(rng: random.Random, number: int) -> Doc:
    sender, recipient = rng.sample(PEOPLE, 2)
    company = rng.choice(COMPANIES)
    deadline = date(2025, rng.randint(2, 11), rng.randint(1, 28))
    deposit = Decimal(rng.randrange(5, 60) * 100)
    topic = rng.choice(SERVICES).lower()
    lines = [
        f"From: {sender}",
        f"To: {recipient}",
        f"Subject: Follow up on {topic}",
        "",
        f"Dear {recipient.split()[0]},",
        "",
        f"Thank you for meeting with us last week. I am writing to follow up on the {topic} "
        f"proposal from {company.split()[0]}.",
        "",
        f"Please find the revised plan attached. To hold the schedule we would need a deposit of "
        f"{_usd(deposit)} and your confirmation by {_long_date(deadline)}.",
        "",
        "Please let me know if anything in the plan needs to change.",
        "",
        "Kind regards,",
        sender,
    ]
    doc_id = f"{number:03d}_letter_{sender.split()[1].lower()}"
    return Doc(
        doc_id=doc_id,
        filename=f"{doc_id}.txt",
        content="\n".join(lines) + "\n",
        expected_type="correspondence",
        golden={
            "intake": {"truncated": False, "high_noise": False},
            "extraction": {
                "parties": sorted([sender, recipient], key=str.casefold),
                "dates": [deadline.isoformat()],
                "amounts": [_key(deposit)],
            },
            "classification": {"doc_type": "correspondence"},
            "summarization": {"doc_type": "correspondence"},
        },
    )


def _memo(rng: random.Random, number: int) -> Doc:
    sender, recipient = rng.sample(PEOPLE, 2)
    lines = [
        "MEMORANDUM",
        "",
        f"From: {sender}",
        f"To: {recipient}",
        "Subject: Warehouse consolidation review",
        "",
        f"Dear {recipient.split()[0]},",
        "",
        "## Executive Summary",
        "Please find below the findings of the warehouse review. The analysis covers the last "
        "two quarters.",
        "",
        "## Findings",
        "- Utilisation at the east site averaged 61 percent.",
        "- Picking errors were concentrated on the night shift.",
        "",
        "## Recommendations",
        "We recommend consolidating the two sites before the peak season.",
        "",
        "Thank you, and kind regards,",
        sender,
    ]
    doc_id = f"{number:03d}_memo_review"
    golden = {
        "intake": {"truncated": False, "high_noise": False},
        "extraction": {"parties": sorted([sender, recipient], key=str.casefold), "dates": [], "amounts": []},
        "classification": {"doc_type": {"any_of": ["report", "correspondence"]}},
    }
    return Doc(
        doc_id=doc_id,
        filename=f"{doc_id}.md",
        content="\n".join(lines) + "\n",
        expected_type="report",
        golden=golden,
        expected_failure="ambiguous_type",
        expected_warning_codes=["ambiguous_classification"],
        notes="A memo header and sign-off wrapped around report sections. Either label is "
        "defensible; the failure is presenting one without saying so.",
    )


_NOISE = "~^|{}<>*=+_\\`"


def _scanned(doc: Doc, rng: random.Random) -> Doc:
    """Damage a clean letter the way a bad scan does."""
    damaged = []
    for char in doc.content:
        if char.isalnum() and rng.random() < 0.42:
            damaged.append(rng.choice(_NOISE))
        else:
            damaged.append(char)
    doc.content = "".join(damaged)
    doc.doc_id = doc.doc_id.replace("_letter_", "_scan_")
    doc.filename = f"{doc.doc_id}.pdf.txt"
    doc.expected_failure = "ocr_damage"
    doc.expected_warning_codes = ["high_noise_ratio"]
    doc.notes = "A fax-quality scan: over a third of the characters are noise."
    doc.golden.pop("extraction", None)
    doc.golden.pop("summarization", None)
    return doc


# -- corpus -------------------------------------------------------------------------


FAILURE_PLAN: dict[int, tuple[str, str]] = {
    4: ("contract_long", "Every date, the fee and the signature sit past the 12k intake budget."),
    9: ("hallucinate_party", "The model invents a signatory, 'John Smith', who is not in the contract."),
    13: ("invent_total", "The summarizer states a total the invoice never gives."),
    17: ("misclassify", "The classifier confidently calls an invoice a report; no check fires."),
    21: ("memo", "Honestly between report and correspondence."),
    24: ("drop_amounts", "The extractor returns no amounts from a fully itemised invoice."),
    28: ("scan", "OCR-damaged letter."),
    31: ("hallucinate_date", "The model invents an effective date of January 1, 2024."),
    35: ("call_timeout", "The classification call times out."),
    38: ("mixed_currency", "USD services plus EUR travel, one USD total presented as the whole."),
    42: ("swap_date_format", "Day-first dates read month-first by the model."),
    45: ("empty", "A blank upload: whitespace only."),
    48: ("contract_long", "Second long contract - truncation is a pattern, not a one-off."),
}

_ROTATION = ("contract", "invoice", "report", "letter")


def build_corpus(seed: int = SEED, size: int = CORPUS_SIZE) -> list[Doc]:
    rng = random.Random(seed)
    docs: list[Doc] = []
    for number in range(1, size + 1):
        plan = FAILURE_PLAN.get(number)
        if plan is None:
            kind = _ROTATION[number % 4]
            builder = {"contract": _contract, "invoice": _invoice, "report": _report, "letter": _letter}[kind]
            docs.append(builder(rng, number))
            continue

        name, note = plan
        if name == "contract_long":
            doc = _contract(rng, number, long_tail=True)
            doc.doc_id = doc.doc_id.replace("_contract_", "_contract_long_")
            doc.filename = f"{doc.doc_id}.md"
            doc.expected_failure = "truncation"
            doc.expected_warning_codes = ["document_truncated"]
        elif name in ("hallucinate_party", "hallucinate_date"):
            doc = _contract(rng, number)
            doc.simulated_faults = [name]
            doc.expected_failure = name
        elif name in ("invent_total", "misclassify", "drop_amounts"):
            doc = _invoice(rng, number)
            doc.simulated_faults = [name]
            doc.expected_failure = name
        elif name == "swap_date_format":
            doc = _invoice(rng, number, numeric_dates=True)
            doc.simulated_faults = [name]
            doc.expected_failure = name
            doc.expected_warning_codes = ["date_normalization_mismatch"]
        elif name == "mixed_currency":
            doc = _invoice(rng, number, second_currency=True)
            doc.expected_failure = "mixed_currency_total"
            doc.expected_warning_codes = ["single_total_over_mixed_currencies"]
            doc.golden.pop("summarization", None)
            doc.golden["extraction"].pop("total", None)
        elif name == "memo":
            doc = _memo(rng, number)
        elif name == "call_timeout":
            doc = _letter(rng, number)
            doc.simulated_faults = [name]
            doc.expected_failure = "llm_timeout"
        elif name == "scan":
            doc = _scanned(_letter(rng, number), rng)
        elif name == "empty":
            doc = Doc(
                doc_id=f"{number:03d}_upload_blank",
                filename=f"{number:03d}_upload_blank.txt",
                content="   \n\n  \t \n",
                expected_type="report",
                golden={},
                expected_failure="empty_document",
                expected_warning_codes=["whitespace_only_document"],
            )
        else:  # pragma: no cover - plan is static
            raise ValueError(name)
        doc.notes = doc.notes or note
        docs.append(doc)
    return docs


def write_corpus(directory: Path | str, seed: int = SEED, size: int = CORPUS_SIZE) -> list[Doc]:
    """Write documents plus ``.meta.json`` and ``.golden.json`` sidecars."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for stale in root.iterdir():
        if stale.is_file():
            stale.unlink()
    docs = build_corpus(seed, size)
    for doc in docs:
        path = root / doc.filename
        path.write_text(doc.content, encoding="utf-8")
        meta = {
            "doc_id": doc.doc_id,
            "expected_type": doc.expected_type,
            "expected_failure": doc.expected_failure,
            "expected_warning_codes": doc.expected_warning_codes,
            "simulated_faults": doc.simulated_faults,
            "notes": doc.notes,
        }
        Path(f"{path}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        golden = {"doc_id": doc.doc_id, "expected": doc.golden, "notes": "Written by the generator."}
        Path(f"{path}.golden.json").write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    return docs
