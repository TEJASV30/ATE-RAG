from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SynonymGroup:
    name: str
    terms: tuple[str, ...]
    boost: float = 0.05


DOMAIN_SYNONYM_GROUPS: tuple[SynonymGroup, ...] = (
    SynonymGroup("issuer_entity", ("issuer", "company", "our company", "registrant", "the company"), 0.07),
    SynonymGroup("legal_counsel", ("legal counsel", "counsel", "law firm", "legal advisor", "legal adviser"), 0.07),
    SynonymGroup("brlm", ("brlm", "brlms", "book running lead manager", "book running lead managers", "lead manager", "lead managers"), 0.07),
    SynonymGroup("agreement", ("agreement", "contract", "arrangement", "understanding", "deed", "instrument"), 0.05),
    SynonymGroup("clause", ("clause", "section", "provision", "article", "paragraph", "term"), 0.05),
    SynonymGroup("termination", ("termination", "terminate", "terminated", "cancellation", "cancel", "expiry", "expiration", "end", "cessation"), 0.06),
    SynonymGroup("effective_date", ("effective date", "commencement date", "start date", "execution date", "date of agreement"), 0.05),
    SynonymGroup("party", ("party", "parties", "counterparty", "client", "customer", "user"), 0.05),
    SynonymGroup("vendor", ("vendor", "supplier", "service provider", "third party", "contractor", "provider"), 0.06),
    SynonymGroup("payment", ("payment", "fee", "fees", "charge", "charges", "consideration", "amount", "price", "compensation"), 0.05),
    SynonymGroup("liability", ("liability", "obligation", "responsibility", "duty", "undertaking", "commitment"), 0.05),
    SynonymGroup("confidentiality", ("confidentiality", "confidential", "non-disclosure", "nondisclosure", "privacy", "secrecy"), 0.05),
    SynonymGroup("audit", ("auditor", "auditors", "statutory auditor", "accountant", "accountants", "independent auditor"), 0.06),
    SynonymGroup("director", ("director", "directors", "board", "board of directors", "management"), 0.04),
    SynonymGroup("shareholder", ("shareholder", "shareholders", "stockholder", "stockholders", "member", "members", "equity holder"), 0.05),
    SynonymGroup("subsidiary", ("subsidiary", "affiliate", "associate", "group company", "holding company"), 0.05),
    SynonymGroup("risk", ("risk", "risks", "risk factor", "risk factors", "threat", "uncertainty", "exposure"), 0.05),
    SynonymGroup("revenue", ("revenue", "revenues", "sales", "income", "turnover", "receipts"), 0.05),
    SynonymGroup("profit", ("profit", "profits", "earnings", "net income", "income", "margin", "margins"), 0.05),
    SynonymGroup("policy", ("policy", "policies", "procedure", "procedures", "guideline", "guidelines", "framework"), 0.04),
    SynonymGroup("security", ("security", "collateral", "charge", "pledge", "hypothecation", "lien"), 0.05),
    SynonymGroup("loan", ("loan", "debt", "borrowing", "facility", "credit facility", "financing"), 0.05),
)


LEGAL_COUNSEL_QUERY_PATTERN = re.compile(r"\blegal\s+counsel\b|\bcounsel\b|\blaw\s+firm\b|\blegal\s+advis[oe]r", re.I)
ISSUER_QUERY_PATTERN = re.compile(r"\bissuer\b|\bcompany\b|\bour\s+company\b|\bregistrant\b", re.I)
BRLM_QUERY_PATTERN = re.compile(r"\bbrlm\b|\bbrlms\b|\bbook\s+running\s+lead\s+manager", re.I)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]*")
TYPO_STOPWORDS = {
    "what",
    "who",
    "where",
    "when",
    "why",
    "how",
    "which",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "in",
    "on",
    "and",
    "or",
    "by",
    "with",
    "from",
    "show",
    "give",
    "tell",
    "me",
}


def _domain_vocabulary() -> set[str]:
    vocabulary: set[str] = set()
    for group in DOMAIN_SYNONYM_GROUPS:
        for term in group.terms:
            for token in TOKEN_RE.findall(term.lower()):
                if len(token) >= 4:
                    vocabulary.add(token)
    vocabulary.update({"legal", "statutory", "issuer", "counsel", "company", "registrant"})
    return vocabulary


DOMAIN_VOCABULARY = _domain_vocabulary()


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.I)


def phrase_present(text: str, phrase: str) -> bool:
    return bool(_phrase_pattern(phrase).search(text))


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 2:
        return 3

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = i
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        previous = current
        if row_min > 2:
            return 3
    return previous[-1]


def _best_domain_correction(token: str) -> str | None:
    lowered = token.lower()
    if lowered in TYPO_STOPWORDS or lowered in DOMAIN_VOCABULARY or len(lowered) < 5:
        return None

    best_term = ""
    best_distance = 3
    for term in DOMAIN_VOCABULARY:
        distance = _edit_distance(lowered, term)
        if distance < best_distance:
            best_term = term
            best_distance = distance

    if not best_term:
        return None
    max_allowed = 1 if len(lowered) <= 5 else 2
    if best_distance <= max_allowed:
        return best_term
    return None


def normalize_domain_typos(query: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        correction = _best_domain_correction(token)
        if correction is None:
            return token
        if token.isupper():
            return correction.upper()
        if token[0].isupper():
            return correction.capitalize()
        return correction

    return TOKEN_RE.sub(replace, query)


def matched_synonym_groups(query: str) -> list[SynonymGroup]:
    query = normalize_domain_typos(query)
    matches: list[SynonymGroup] = []
    for group in DOMAIN_SYNONYM_GROUPS:
        if any(phrase_present(query, term) for term in group.terms):
            matches.append(group)
    return matches


def _matched_terms_for_group(query: str, group: SynonymGroup) -> list[str]:
    # Prefer longer phrases so "statutory auditor" wins over the nested "auditor".
    matches: list[str] = []
    occupied_spans: list[tuple[int, int]] = []
    for term in sorted(group.terms, key=len, reverse=True):
        for match in _phrase_pattern(term).finditer(query):
            span = match.span()
            if any(not (span[1] <= used[0] or span[0] >= used[1]) for used in occupied_spans):
                continue
            matches.append(term)
            occupied_spans.append(span)
            break
    return matches


def _clean_variant(query: str) -> str:
    query = re.sub(r"\b(the|a|an)\s+our\s+company\b", "our company", query, flags=re.I)
    query = re.sub(r"\b(the|a|an)\s+(the\s+)", r"\2", query, flags=re.I)
    query = re.sub(r"\bstatutory\s+statutory\s+auditor\b", "statutory auditor", query, flags=re.I)
    query = re.sub(r"\s+", " ", query)
    return query.strip()


def _replace_phrase_once(query: str, source: str, target: str) -> str:
    return _clean_variant(_phrase_pattern(source).sub(target, query, count=1))


def expanded_queries(query: str, *, max_variants: int = 8) -> list[str]:
    variants = [query]
    normalized_query = normalize_domain_typos(query)
    if normalized_query != query:
        variants.append(normalized_query)
    additions: list[str] = []

    for group in matched_synonym_groups(normalized_query):
        matched_terms = _matched_terms_for_group(normalized_query, group)
        additions.extend(term for term in group.terms if term not in matched_terms)

        for matched_term in matched_terms:
            for alternate in group.terms:
                if alternate == matched_term:
                    continue
                variants.append(_replace_phrase_once(normalized_query, matched_term, alternate))
                if len(variants) >= max_variants:
                    break
            if len(variants) >= max_variants:
                break
        if len(variants) >= max_variants:
            break

    if additions:
        variants.append(f"{normalized_query} {' '.join(additions[:24])}")

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        normalized = " ".join(variant.split())
        key = normalized.lower()
        if normalized and key not in seen:
            deduped.append(normalized)
            seen.add(key)
    return deduped[:max_variants]


def expanded_query_for_rerank(query: str) -> str:
    normalized_query = normalize_domain_typos(query)
    groups = matched_synonym_groups(normalized_query)
    if not groups:
        return normalized_query
    synonym_text = "; ".join(f"{group.name}: {', '.join(group.terms)}" for group in groups[:4])
    return f"{normalized_query}\nRelevant equivalent terms: {synonym_text}"


def synonym_text_boost(query: str, text: str) -> float:
    query = normalize_domain_typos(query)
    boost = 0.0
    for group in matched_synonym_groups(query):
        query_terms = [term for term in group.terms if phrase_present(query, term)]
        text_terms = [term for term in group.terms if phrase_present(text, term)]
        if text_terms:
            boost += group.boost
            if query_terms and not any(term in text_terms for term in query_terms):
                boost += min(group.boost, 0.04)
    return min(boost, 0.22)


def role_specific_boost(query: str, text: str) -> float:
    query = normalize_domain_typos(query)
    text_lower = text.lower()
    boost = 0.0
    if LEGAL_COUNSEL_QUERY_PATTERN.search(query):
        if re.search(r"legal\s+counsel\s+to\s+(our\s+)?company", text_lower):
            boost += 0.32
        if re.search(r"legal\s+counsel\s+to\s+(the\s+)?issuer", text_lower):
            boost += 0.32
        if ISSUER_QUERY_PATTERN.search(query) and re.search(r"legal\s+counsel\s+to\s+(the\s+)?brlms?", text_lower):
            boost -= 0.22
        if ISSUER_QUERY_PATTERN.search(query) and "book running lead manager" in text_lower:
            boost -= 0.08
        if BRLM_QUERY_PATTERN.search(query) and re.search(r"legal\s+counsel\s+to\s+(the\s+)?brlms?", text_lower):
            boost += 0.32
    return boost
