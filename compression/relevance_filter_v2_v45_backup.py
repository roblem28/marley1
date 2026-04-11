#!/usr/bin/env python3
"""Relevance Filter v4.5 -- hybrid-best build.

v4.5 hybrid-best patches over v4.4:
A. Metering: +4.0 monitoring lines, +2.0 UFC 3-550-01, -4.0 cyber unless monitoring
B. Islanding: +4.0 behavioral, -3.0 bare headers, -3.0 compliance-only, no standalone headers
C. C5ISR power+grounding: drop clipped Motorola R56, tighter aspect bullets
D. Frozen families preserved: progressive collapse, blast, cybersecurity, laundry, closet, renewable, antenna/RF
E. New output fields: proof_quality_score, family_handler_version
"""

import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


SECTION_PATTERNS = [
    re.compile(r'^([A-Z]?-?\d{1,2}-\d{1,2}(?:\.\d{1,3})*)\s+(.+)', re.MULTILINE),
    re.compile(r'^(CHAPTER\s+\d+)\s*(.*)', re.MULTILINE),
    re.compile(r'^(APPENDIX\s+[A-Z])\s*(.*)', re.MULTILINE),
    re.compile(r'^((?:TABLE|FIGURE)\s+[A-Z]?-?\d+-\d+)\s*(.*)', re.MULTILINE),
]
_UNIT_RE = re.compile(
    r'\b(?:NSF|sf|SF|gsf|GSF|kW|MW|kVA|MVA|BTU|CFM|GPM|psi|PSI|pCi/L|dB|dBA|'
    r'lux|fc|ft|feet|inches|in\.|meters|m\b|mm|gallons|lbs|tons|percent|%|'
    r'amps?|volts?|watts?|ohms?|Hz|hours?|minutes?|seconds?)\b')
_REQUIREMENT_WORDS_RE = re.compile(r'\b(?:shall|must|required|comply\s+with|refer\s+to|in\s+accordance\s+with)\b', re.I)
_PROVIDE_WITH_VALUE_RE = re.compile(r'\bprovide\b.*?(?:\d|' + _UNIT_RE.pattern + r'|UFC|ASCE|ACI|IEEE|NFPA|ASTM|ASHRAE)', re.I)
_MIN_MAX_WITH_VALUE_RE = re.compile(r'\b(?:minimum|maximum)\b.*?(?:\d|' + _UNIT_RE.pattern + r'|UFC|ASCE|ACI|IEEE|NFPA|ASTM|ASHRAE)', re.I)
_XREF_RE = re.compile(r'(?:Table|Section|UFC|ASCE|ACI|IEEE|NFPA|ASTM|ASHRAE)\s+[\dA-Z]', re.I)
_SECTION_HEADER_LINE_RE = re.compile(r'^[A-Z]?-?\d{1,2}-\d{1,2}(?:\.\d{1,3})*\s+', re.MULTILINE)
_CHAPTER_HEADER_RE = re.compile(r'^(?:CHAPTER\s+\d+|APPENDIX\s+[A-Z])', re.MULTILINE)
_TABLE_ROW_PARSED_RE = re.compile(r'^\[TABLE\]\s|^Table\s+\S+\s*[\|(]')
_TAGGED_LINE_RE = re.compile(r'\b(?:exception|note|replacement|supplement|addition|deletion)\b', re.I)
_NUMBER_RE = re.compile(r'\d+(?:\.\d+)?')
_STANDARD_RE = re.compile(r'\b(?:UFC|ASCE|ACI|IEEE|NFPA|ASTM|ASHRAE|IBC|IEC|ANSI|MIL-STD|MIL-HDBK|UFGS|ETL)\s*[\d-]', re.I)
_SECTION_REF_RE = re.compile(r'(?:Section|Table|Figure|Appendix|Chapter)\s+[\dA-Z]', re.I)
_GOVERNING_LINE_RE = re.compile(r'\b(?:apply|comply\s+with|refer\s+to|use|in\s+accordance\s+with)\s+(?:UFC|ASCE|ACI|IEEE|NFPA|ASTM|ASHRAE|MIL-STD)\s*[\d-]', re.I)
_TOC_LINE_RE = re.compile(r'\.{2,}\s*\d+\s*$')


# ==================================================================
# QUERY FAMILY DETECTION (v4.5)
# ==================================================================
def detect_family(query):
    q = query.lower()
    if any(w in q for w in ["metering", "monitoring", "meter", "remote monitoring", "monitor and control"]):
        return "metering"
    if any(w in q for w in ["islanding", "islanded", "black start", "off-grid", "soft transition", "secure islanding"]):
        return "islanding"
    if "power" in q and "grounding" in q and "c5isr" in q:
        return "c5isr_power_grounding"
    if ("antenna" in q and "rf" in q) or ("antenna" in q and "system" in q) or ("rf system" in q):
        return "antenna_rf"
    if any(w in q for w in ["closet", "wardrobe", "hanging space"]) or ("storage" in q and "space" in q):
        return "closet_storage"
    return None


# ==================================================================
# CLIPPED-LINE FILTER (v4.4 new)
# ==================================================================
def _is_clipped(line):
    """Reject lines that are obviously mid-sentence fragments."""
    s = line.strip()
    if not s or len(s) < 8:
        return True
    # Starts with lowercase without being a table row or header
    if s[0].islower() and not _TABLE_ROW_PARSED_RE.match(s) and not _SECTION_HEADER_LINE_RE.match(s):
        return True
    # Very short and no verb/subject signal
    words = s.split()
    if len(words) <= 3 and not any(w.lower() in ("shall","must","required","provide","comply") for w in words):
        return True
    # Ends without punctuation or closing structure (likely truncated)
    if not s[-1] in '.!?):"' and len(words) <= 5:
        return True
    return False


def _filter_clipped(lines):
    """Remove clipped lines, return (kept, rejected)."""
    kept, rejected = [], []
    for l in lines:
        if _is_clipped(l):
            rejected.append(l)
        else:
            kept.append(l)
    return kept, rejected


# ==================================================================
# PDF EXTRACTION (unchanged from v4.3)
# ==================================================================
def _is_toc_page(pt):
    lines = pt.strip().split('\n')
    if len(lines) < 3: return False
    return sum(1 for l in lines if _TOC_LINE_RE.search(l)) / len(lines) > 0.5

def _find_toc_range(pages):
    ts, te = None, None
    for i, t in enumerate(pages):
        if ts is None:
            if 'TABLE OF CONTENTS' in t.upper(): ts = i
        elif te is None:
            if not _is_toc_page(t):
                lines = t.strip().split('\n')
                first = lines[0].strip() if lines else ""
                if re.match(r'^(CHAPTER\s+\d+|[A-Z]?-?\d{1,2}-\d{1,2})', first) and '...' not in first:
                    te = i; break
    if ts is not None and te is None:
        for i in range(ts, len(pages)):
            if not _is_toc_page(pages[i]) and i > ts: te = i; break
    return ts, te

def _normalize_header(h):
    return re.sub(r'\s+', ' ', re.sub(r'\.{2,}\s*\d+\s*$', '', h)).strip().lower()

def extract_text_no_toc(pdf_path):
    import pdfplumber
    pdf = pdfplumber.open(pdf_path)
    all_p = [p.extract_text() or "" for p in pdf.pages]; pdf.close()
    ts, te = _find_toc_range(all_p)
    skip, kept = [], []
    for i, t in enumerate(all_p):
        if ts is not None and te is not None and ts <= i < te: skip.append(i)
        elif _is_toc_page(t): skip.append(i)
        else: kept.append(t)
    return '\n'.join(kept), skip

def extract_pdf_tables(pdf_path):
    import pdfplumber
    from collections import defaultdict
    pdf = pdfplumber.open(pdf_path); chunks = []; ctn = None; ctt = ""
    for pi, page in enumerate(pdf.pages):
        pt = page.extract_text() or ""
        if _is_toc_page(pt): continue
        labels = re.findall(r'(?:Table|TABLE)\s+([A-Z]?-?\d+-\d+(?:\.\d+)?)\s*(.*?)(?:\n|$)', pt)
        if labels: ctn = labels[-1][0]; ctt = labels[-1][1].strip()
        pfx = f"[TABLE] Table {ctn}" if ctn else "[TABLE] Table"
        if ctt: pfx = f"{pfx} ({ctt})"
        tables = page.extract_tables()
        if tables:
            for table in tables:
                if not table or len(table) < 2: continue
                ch, ds = [], 0
                for ri, row in enumerate(table):
                    cells = [str(c).strip().replace('\n',' ') if c else "" for c in row]
                    ne = [c for c in cells if c]
                    if not ne: continue
                    if sum(1 for c in ne if re.match(r'^[\d,.]+$', c)) <= 1 and len(ne) >= 2:
                        ch = cells; ds = ri+1; break
                if not ch and table: ch = [str(c).strip().replace('\n',' ') if c else "" for c in table[0]]; ds = 1
                for row in table[ds:]:
                    cells = [str(c).strip().replace('\n',' ') if c else "" for c in row]
                    if not any(c for c in cells if c): continue
                    parts = []
                    for ci, val in enumerate(cells):
                        if not val: continue
                        if ci < len(ch) and ch[ci]: parts.append(f"{ch[ci]}: {val}")
                        else: parts.append(val)
                    if parts: chunks.append(f"{pfx} | {' | '.join(parts)}")
        elif 'Table' in pt or 'TABLE' in pt:
            words = page.extract_words()
            if not words: continue
            rby = defaultdict(list)
            for w in words: rby[round(w["top"],0)].append(w)
            for y in sorted(rby):
                rw = sorted(rby[y], key=lambda w: w["x0"])
                xp = [w["x0"] for w in rw]
                if max(xp)-min(xp) < 150 or len(rw) < 3: continue
                if not any(re.search(r'\d', w["text"]) for w in rw): continue
                chunks.append(f"{pfx} | {' | '.join(w['text'] for w in rw)}")
    pdf.close(); return chunks


class RelevanceFilterV4:
    def __init__(self, bi_model="all-MiniLM-L6-v2", cross_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.bi_model = SentenceTransformer(bi_model)
        self.cross_model = CrossEncoder(cross_model)

    # CHUNKING (unchanged)
    def chunk_with_headers(self, text, chunk_size=None, pdf_tables=None):
        secs, hdrs = self._split_sections(text, return_headers=True)
        cs, ch = [], []
        for s, h in zip(secs, hdrs):
            if _TOC_LINE_RE.search(h): continue
            cs.append(s); ch.append(h)
        seen = {}; fs, fh = [], []
        for s, h in zip(cs, ch):
            n = _normalize_header(h)
            if n in seen:
                if len(s) > len(fs[seen[n]]): fs[seen[n]] = s
            else: seen[n] = len(fs); fs.append(s); fh.append(h)
        if pdf_tables:
            for t in pdf_tables:
                m = re.match(r'\[TABLE\]\s*(Table\s+\S+)', t)
                fs.append(t); fh.append(m.group(1) if m else "Table")
        if not fs:
            c = self._chunk_sentences(text, chunk_size or 5)
            return c, [""]*len(c)
        return fs, fh

    def _split_sections(self, text, return_headers=False):
        hf = []
        for pat in SECTION_PATTERNS:
            for m in pat.finditer(text):
                hf.append((m.start(), m.end(), f"{m.group(1).strip()} {(m.group(2) or '').strip()}".strip()))
        if not hf: return ([], []) if return_headers else []
        hf.sort(key=lambda x: x[0])
        dd = [hf[0]]
        for h in hf[1:]:
            if h[0] >= dd[-1][1]: dd.append(h)
        cs, hs = [], []
        for i, (s, _, ht) in enumerate(dd):
            e = dd[i+1][0] if i+1 < len(dd) else len(text)
            st = text[s:e].strip()
            if st and len(st.split()) >= 5: cs.append(st); hs.append(ht)
        return (cs, hs) if return_headers else cs

    def _chunk_sentences(self, t, cs):
        ss = [s for s in re.split(r'(?<=[.!?])\s+', t.strip()) if s.strip()]
        return [" ".join(ss[i:i+cs]) for i in range(0, len(ss), cs)]

    # PRECOMPUTE
    def precompute(self, chunks):
        self._cached_chunks = chunks
        self._cached_embs = self.bi_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        self._bm25 = BM25Okapi([c.lower().split() for c in chunks])
        return self._cached_embs, self._bm25

    def _header_boost(self, q, hdrs, n):
        b = np.zeros(n)
        if not hdrs: return b
        qw = set(re.findall(r'[a-z]+', q.lower())) - {"what","are","the","for","of","in","a","an","and","or","how","should","be","is","to","do","requirements"}
        for i, h in enumerate(hdrs):
            if i >= n: break
            if qw & set(re.findall(r'[a-z]+', h.lower())): b[i] = 1.0
        return b

    # TWO-LANE RETRIEVAL (unchanged)
    def filter_two_lane(self, lq, eq, top_k=10, headers=None, chunk_embs=None, chunks=None, bm25_index=None, answer_mode="requirement_list", cap_expansion=False):
        if chunk_embs is None: chunk_embs = self._cached_embs
        if chunks is None: chunks = self._cached_chunks
        bm25 = bm25_index if bm25_index is not None else self._bm25
        n = len(chunks)
        def _sc(q):
            qe = self.bi_model.encode([q], convert_to_numpy=True)[0]
            nm = np.linalg.norm(chunk_embs, axis=1)*np.linalg.norm(qe)
            nm = np.where(nm==0,1e-10,nm)
            d = np.dot(chunk_embs,qe)/nm
            rb = bm25.get_scores(q.lower().split())
            mx = rb.max() if rb.max()>0 else 1.0
            return 0.5*d + 0.3*(rb/mx) + 0.2*self._header_boost(q,headers,n), d, rb/mx
        fl, dl, sl = _sc(lq)
        lt = list(np.argsort(fl)[::-1][:top_k])
        if eq != lq:
            fe, _, _ = _sc(eq)
            if cap_expansion: fe *= 0.10
            et = list(np.argsort(fe)[::-1][:top_k])
        else: et = []
        merged = list(dict.fromkeys(lt + et))
        if answer_mode in ("governing_reference","numeric_lookup") and headers:
            qw = set(re.findall(r'[a-z]+', lq.lower())) - {"what","are","the","for","of","in","a","an","and","or","how","should","be","is","to","do","requirements"}
            pinned = [i for i in range(min(n,len(headers))) if len(qw & set(re.findall(r'[a-z]+', headers[i].lower()))) >= 2]
            for p in reversed(pinned[:2]):
                if p in merged: merged.remove(p)
                merged.insert(0, p)
        pool = merged[:top_k*3]
        pairs = [[lq, chunks[int(i)]] for i in pool]
        rp = self.cross_model.predict(pairs)
        scored = sorted(zip(pool, rp), key=lambda x: x[1], reverse=True)[:top_k]
        return [int(i) for i,_ in scored], dl, sl, {int(i): float(s) for i,s in scored}

    # EVIDENCE EXTRACTION (unchanged)
    def extract_evidence(self, text):
        lines = text.split('\n'); kept = []; ph = False
        for line in lines:
            s = line.strip()
            if not s: ph = False; continue
            if _TABLE_ROW_PARSED_RE.match(s): kept.append(s); ph = False; continue
            if _SECTION_HEADER_LINE_RE.match(s) or _CHAPTER_HEADER_RE.match(s): kept.append(s); ph = True; continue
            if ph:
                ss = re.split(r'(?<=[.!?])\s+', s)
                if ss: kept.append(ss[0])
                ph = False; continue
            ph = False
            if _TAGGED_LINE_RE.search(s): kept.append(s); continue
            ks = []
            for sent in re.split(r'(?<=[.!?])\s+', s):
                if _REQUIREMENT_WORDS_RE.search(sent): ks.append(sent)
                elif _XREF_RE.search(sent): ks.append(sent)
                elif re.search(r'\bprovide\b', sent, re.I) and _PROVIDE_WITH_VALUE_RE.search(sent): ks.append(sent)
                elif re.search(r'\b(?:minimum|maximum)\b', sent, re.I) and _MIN_MAX_WITH_VALUE_RE.search(sent): ks.append(sent)
            if ks: kept.append(' '.join(ks))
        return '\n'.join(kept)

    # ANSWERABILITY (unchanged)
    def score_answerability(self, evidence, query):
        r = {"has_governing_ref":False,"has_table_row":False,"has_requirement":False,"has_numeric":False,"missing":[]}
        if not evidence or not evidence.strip():
            r["coverage"]="unsupported"; r["missing"]=["no evidence found"]; return r
        for l in evidence.strip().split('\n'):
            if _TABLE_ROW_PARSED_RE.match(l): r["has_table_row"]=True
            if _STANDARD_RE.search(l): r["has_governing_ref"]=True
            if _REQUIREMENT_WORDS_RE.search(l): r["has_requirement"]=True
            if _NUMBER_RE.search(l) and _UNIT_RE.search(l): r["has_numeric"]=True
        sc = sum([r["has_governing_ref"],r["has_table_row"],r["has_requirement"],r["has_numeric"]])
        r["coverage"] = "supported" if sc>=2 else "partially_supported" if sc>=1 else "unsupported"
        for k,lb in [("has_governing_ref","governing standard reference"),("has_requirement","requirement sentence"),
                     ("has_numeric","numeric value with unit"),("has_table_row","table data")]:
            if not r[k]: r["missing"].append(lb)
        return r

    # ==================================================================
    # FAMILY FINALIZERS (v4.5 -- hybrid-best)
    # ==================================================================

    def finalize_metering(self, evidence, query):
        """v4.5 Patch A: +4.0 monitoring lines, +2.0 UFC 3-550-01, -4.0 cyber unless monitoring.
        Joins continuation lines before scoring to avoid clipping monitoring content."""
        raw_lines = [l.strip() for l in evidence.strip().split('\n') if l.strip()]

        # Join continuation lines: if a line starts lowercase and previous line
        # doesn't end with period, merge them into one sentence
        joined = []
        for l in raw_lines:
            if joined and l and l[0].islower() and not joined[-1].rstrip().endswith(('.', ')', '"')):
                joined[-1] = joined[-1].rstrip() + ' ' + l
            else:
                joined.append(l)

        # Now split into sentences at period boundaries for finer scoring
        sentences = []
        for block in joined:
            for sent in re.split(r'(?<=[.!?])\s+', block):
                s = sent.strip()
                if s and len(s) > 10:
                    sentences.append(s)

        scored = []
        for l in sentences:
            ll = l.lower()
            s = 0.0
            # Strong monitoring signal (+4.0) - coordinate terms only count if line has monitoring/metering context
            if any(w in ll for w in ["remote monitoring", "monitoring and control",
                                      "metering requirements", "metering and monitoring"]): s += 4.0
            elif any(w in ll for w in ["coordinate with the activity", "coordinate with the government",
                                        "coordinated with the activity", "control requirements"]):
                if any(w in ll for w in ["monitoring", "metering", "monitor", "meter"]): s += 4.0
                else: s += 1.0  # coordination without monitoring context is weak signal
            # Distribution/compliance signal (+2.0)
            if re.search(r'comply\s+with\s+UFC\s+3-550', ll) or "exterior distribution" in ll: s += 2.0
            # Hard penalty for cybersecurity center of gravity (-4.0)
            if any(w in ll for w in ["ufc 4-010-06", "cybersecurity", "facility-related control systems"]):
                if not any(w in ll for w in ["remote monitoring", "monitoring and control", "metering",
                                              "monitor"]): s -= 4.0
            # Penalty for telecom/non-monitoring content
            if any(w in ll for w in ["telecommunications", "telecom"]) and not any(w in ll for w in ["monitoring", "metering"]): s -= 2.0
            scored.append((l, s))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build max 2 sentences, monitoring-centered
        found = []
        for l, s in scored:
            if s >= 2.0 and len(found) < 2:
                # Cannot lead with cybersecurity (exempt if line also has monitoring language)
                if not found and any(w in l.lower() for w in ["cybersecurity", "ufc 4-010-06"]):
                    if not any(w in l.lower() for w in ["monitoring", "metering", "monitor"]):
                        continue
                found.append(l)
        if not found:
            # Fallback: best non-cyber, non-telecom requirement line
            for l, s in scored:
                if s >= 0 and _REQUIREMENT_WORDS_RE.search(l):
                    ll = l.lower()
                    if not any(w in ll for w in ["cybersecurity", "ufc 4-010-06", "telecommunications"]):
                        found.append(l)
                        break

        if found:
            parts = ". ".join(f.rstrip('.') for f in found[:2])
            return f"The retained text specifies: {parts}. It does not provide a complete metering requirement set in the retained sections."
        return "The retained text does not specify metering requirements in the retained sections."

    def finalize_islanding(self, evidence, query):
        """v4.5 Patch B: +4.0 behavioral, -3.0 bare headers, -3.0 compliance-only, no standalone headers."""
        lines = [l.strip() for l in evidence.strip().split('\n') if l.strip()]
        lines, _ = _filter_clipped(lines)

        # Detect bare headers (section number + title, no requirement verb)
        def _is_bare_header(line):
            if _SECTION_HEADER_LINE_RE.match(line) or _CHAPTER_HEADER_RE.match(line):
                if not _REQUIREMENT_WORDS_RE.search(line):
                    return True
            return False

        scored = []
        for l in lines:
            ll = l.lower()
            s = 0.0
            # Behavioral line boosts (+4.0)
            if any(w in ll for w in ["transitioning to islanded power", "transition to islanded power",
                                      "deliver power to designated critical loads",
                                      "deliver power to designated critical",
                                      "black start", "grid-forming",
                                      "off-grid endurance", "off-grid system endurance",
                                      "secure islanding", "soft transition",
                                      "states of operation"]): s += 4.0
            # Secondary behavioral signals
            if any(w in ll for w in ["responsible for transitioning", "critical loads",
                                      "critical mission", "reference frequency",
                                      "restoration time", "disparate source",
                                      "grid-forming der", "minimum of one"]): s += 2.0
            # Bare header penalty (-3.0)
            if _is_bare_header(l): s -= 3.0
            # Compliance-only penalty (-3.0)
            if re.search(r'commercial\s+utility\s+relations', ll) and s < 2.0: s -= 3.0
            if re.search(r'ieee\s+1547|ul\s+1741', ll):
                if not any(w in ll for w in ["transitioning", "black start", "islanded",
                                              "critical load", "grid-forming"]): s -= 3.0
            # Requirement signal bonus
            if _REQUIREMENT_WORDS_RE.search(l): s += 0.5
            scored.append((l, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Build max 4 bullets: (a) transition/critical loads, (b) black start/grid-forming,
        # (c) off-grid endurance, (d) secure islanding/soft transition
        concepts = {
            "transition": ["transition to islanded", "transitioning to islanded",
                           "responsible for transitioning", "critical load", "deliver power"],
            "blackstart": ["black start", "grid-forming"],
            "endurance": ["off-grid endurance", "off-grid system endurance", "endurance not less"],
            "secure": ["secure islanding", "states of operation", "soft transition",
                       "restoration time", "microgrid formation"],
        }
        kept = []
        used_concepts = set()
        for l, s in scored:
            if s <= 0 and len(kept) >= 2: break
            if len(kept) >= 4: break
            # Never emit bare headers as standalone bullets
            if _is_bare_header(l): continue
            ll = l.lower()
            matched_concept = None
            for cname, keywords in concepts.items():
                if any(kw in ll for kw in keywords):
                    matched_concept = cname
                    break
            if matched_concept and matched_concept in used_concepts:
                continue
            if matched_concept:
                used_concepts.add(matched_concept)
            kept.append(l)

        return '\n'.join(kept) if kept else None

    def finalize_antenna_rf(self, evidence, query):
        """v4.4 Patch C: forced 2-aspect split for antenna + RF isolation."""
        lines = [l.strip() for l in evidence.strip().split('\n') if l.strip()]
        lines, _ = _filter_clipped(lines)

        # Aspect 1: Antenna systems / entry / sizing
        antenna_scored = []
        for l in lines:
            ll = l.lower()
            s = 0.0
            if any(w in ll for w in ["cable entry port", "sized according to the cables", "antenna systems",
                                      "cable boots", "entry port", "antenna system"]): s += 3.0
            if any(w in ll for w in ["weatherproof", "compression", "power/trench"]): s += 1.5
            if "hvac" in ll or "temperature" in ll or "telecom grade" in ll: s -= 2.0
            antenna_scored.append((l, s))
        antenna_scored.sort(key=lambda x: x[1], reverse=True)
        antenna_best = [l for l, s in antenna_scored if s >= 1.5][:2]

        # Aspect 2: RF isolation / separation / EMC
        rf_scored = []
        for l in lines:
            ll = l.lower()
            s = 0.0
            if any(w in ll for w in ["isolation", "separation distance", "receiver site",
                                      "rf isolation", "emi", "emc", "electromagnetic"]): s += 3.0
            if any(w in ll for w in ["optimum radio communications", "specific requirements for each project",
                                      "minimum separation", "table 2-1"]): s += 2.0
            if _TABLE_ROW_PARSED_RE.match(l) and any(w in ll for w in ["separation", "mile", "km", "receiver"]): s += 2.5
            if "hvac" in ll or "temperature" in ll or "grade" in ll: s -= 2.0
            rf_scored.append((l, s))
        rf_scored.sort(key=lambda x: x[1], reverse=True)
        rf_best = [l for l, s in rf_scored if s >= 1.5][:2]

        parts = []
        if antenna_best:
            parts.append("Antenna systems: " + ' '.join(antenna_best))
        if rf_best:
            parts.append("RF isolation / separation: " + ' '.join(rf_best))

        if parts:
            return '\n'.join(parts)
        return None

    def finalize_c5isr_power_grounding(self, evidence, query, headers=None, chunks=None, top_indices=None):
        """v4.5 Patch C: tighter aspect bullets, drop clipped Motorola R56."""
        lines = [l.strip() for l in evidence.strip().split('\n') if l.strip()]
        lines, _ = _filter_clipped(lines)

        # Grounding: prefer NFPA 70, TIA-607-D, MIL-STD-188-124B. Drop Motorola R56 if clipped.
        grounding = []
        for l in lines:
            ll = l.lower()
            if any(w in ll for w in ["grounding", "bonding", "equipotential", "earth electrode",
                                      "nfpa 70", "tia-607", "mil-std-188-124", "5 ohm"]):
                if any(w in ll for w in ["temperature", "systems manual", "clean agent"]) and "grounding" not in ll: continue
                # Drop Motorola R56 if line ends abruptly (clipped)
                if "motorola r56" in ll and not l.rstrip().endswith((".", ")", '"')):
                    continue
                grounding.append(l)

        # Power: prefer facility Grade requirement, backup power/generators
        power = []
        for l in lines:
            ll = l.lower()
            if any(w in ll for w in ["grade requirement", "facility grade",
                                      "backup power", "generators", "generator"]):
                if l not in grounding: power.append(l)
        # Fallback: distribution architecture, black start, UPS
        if not power:
            for l in lines:
                ll = l.lower()
                if any(w in ll for w in ["distribution architecture", "black start",
                                          "ups", "dc power", "power distribution"]):
                    if l not in grounding: power.append(l)

        parts = []
        if power: parts.append("Power: " + ' '.join(power[:2]))
        if grounding: parts.append("Grounding and Bonding: " + ' '.join(grounding[:2]))
        if parts: return '\n'.join(parts)
        for l in lines:
            if re.search(r'NFPA\s+70|TIA-607|MIL-STD-188-124', l, re.I):
                return f"Grounding and Bonding: {l}"
        return None

    def finalize_closet_storage(self, evidence, query):
        lines = [l.strip() for l in evidence.strip().split('\n') if l.strip()]
        table_rows = [l for l in lines if _TABLE_ROW_PARSED_RE.match(l)]
        non_table = [l for l in lines if not _TABLE_ROW_PARSED_RE.match(l)]
        scored = []
        for r in table_rows:
            rl = r.lower(); s = 0.0
            if "closet" in rl: s += 4.0
            if re.search(r'closet.*nsf|nsf.*closet|\bcloset\b.*\d', rl): s += 2.0
            if "storage" in rl and "closet" not in rl: s -= 2.0
            if any(w in rl for w in ["kitchenette","bathroom","multipurpose","security"]): s -= 3.0
            scored.append((r, s))
        for l in non_table:
            ll = l.lower()
            if "closet" in ll and (_NUMBER_RE.search(l) or _UNIT_RE.search(l)): scored.append((l, 3.0))
            elif "closet" in ll and _REQUIREMENT_WORDS_RE.search(l): scored.append((l, 2.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] > 0:
            kept = [scored[0][0]]
            for r, s in scored[1:]:
                if _TAGGED_LINE_RE.search(r) and s >= 0: kept.append(r); break
            return '\n'.join(kept)
        return None

    # ==================================================================
    # DETERMINISTIC ANSWERER v4.5
    # ==================================================================
    def deterministic_answer(self, answer_mode, evidence, query, family=None,
                             headers=None, chunks=None, top_indices=None):
        if not evidence or not evidence.strip():
            return None, False, "no_evidence", family, []

        # Family overrides
        if family == "metering":
            ans = self.finalize_metering(evidence, query)
            if ans: return ans, True, "metering_partial_support", family, []
        if family == "islanding":
            ans = self.finalize_islanding(evidence, query)
            if ans: return ans, True, "islanding_behavioral", family, []
        if family == "antenna_rf":
            ans = self.finalize_antenna_rf(evidence, query)
            if ans: return ans, True, "antenna_rf_aspect_split", family, []
        if family == "c5isr_power_grounding":
            ans = self.finalize_c5isr_power_grounding(evidence, query, headers, chunks, top_indices)
            if ans: return ans, True, "c5isr_aspect_split", family, []
        if family == "closet_storage":
            ans = self.finalize_closet_storage(evidence, query)
            if ans: return ans, True, "closet_purified_row", family, []

        # Default
        lines = [l.strip() for l in evidence.strip().split('\n') if l.strip()]
        lines, clipped = _filter_clipped(lines)

        if answer_mode == "governing_reference":
            gov = [l for l in lines if _GOVERNING_LINE_RE.search(l)]
            if gov:
                best = min(gov, key=lambda l: (
                    0 if re.search(r'\bapply\s+UFC', l, re.I) else
                    1 if re.search(r'\bcomply\s+with\s+UFC', l, re.I) else
                    2 if re.search(r'\brefer\s+to\s+UFC', l, re.I) else 3, len(l)))
                return best.strip(), True, "single_governing_line", None, clipped

        if answer_mode == "numeric_lookup":
            trows = [l for l in lines if _TABLE_ROW_PARSED_RE.match(l)]
            numlines = [l for l in lines if not _TABLE_ROW_PARSED_RE.match(l) and _NUMBER_RE.search(l) and _UNIT_RE.search(l)]
            cands = trows or numlines
            if cands:
                kept, _ = self._filter_rows_purity(cands, query)
                return '\n'.join(kept), True, "purified_row", None, clipped

        if answer_mode in ("requirement_list", "partial_support"):
            req = [l for l in lines if _REQUIREMENT_WORDS_RE.search(l) or _GOVERNING_LINE_RE.search(l)]
            if req:
                seen = set(); dd = []
                for l in req:
                    k = l[:40].lower()
                    if k not in seen: seen.add(k); dd.append(l)
                return '\n'.join(dd[:4]), True, "requirement_list_capped", None, clipped

        return None, False, "no_match", None, clipped

    def _filter_rows_purity(self, rows, query):
        scored = [(r, self._score_row(r, query)) for r in rows]
        scored.sort(key=lambda x: x[1], reverse=True)
        kept = [scored[0][0]] if scored else []
        for r, s in scored[1:]:
            if _TAGGED_LINE_RE.search(r) and s > 0: kept.append(r); break
        return kept, {r: s for r, s in scored}

    def _score_row(self, row, query):
        qw = set(re.findall(r'[a-z]+', query.lower())) - {"what","are","the","for","of","in","a","an","and","or","how","should","be","is","to","do","requirements","facility"}
        rw = set(re.findall(r'[a-z]+', row.lower()))
        s = len(qw & rw) * 2.0
        if _NUMBER_RE.search(row) and _UNIT_RE.search(row): s += 1.0
        bad = {"bathroom","kitchenette","multipurpose","security","environmental","comfort","parking","recreation","vestibule","corridor"}
        rl = set(re.findall(r'[a-z]+', row.lower().split('|')[1] if '|' in row else row.lower()))
        if (rl & bad) and not (qw & rl): s -= 2.0
        return s

    # EXTRACTIVE FALLBACK
    def extractive_fallback(self, evidence, query, answer_mode, family=None, headers=None, chunks=None, top_indices=None):
        if not evidence or not evidence.strip(): return None, "no_evidence"
        det, used, reason, fam, _ = self.deterministic_answer(answer_mode, evidence, query, family, headers, chunks, top_indices)
        if used and det: return det, f"deterministic_fallback:{reason}"
        lines = [l.strip() for l in evidence.strip().split('\n') if l.strip()]
        strong = [l for l in lines if _GOVERNING_LINE_RE.search(l) or _TABLE_ROW_PARSED_RE.match(l) or _REQUIREMENT_WORDS_RE.search(l)]
        if strong: return '\n'.join(strong[:4]), "extractive_bullets"
        return '\n'.join(lines[:3]), "raw_evidence"

    # VERIFIER (unchanged)
    def verify_answer(self, answer, evidence):
        if not answer or not evidence: return answer or "Not specified in the retrieved sections."
        words = answer.split()
        if len(words) > 24:
            ngc = {}
            for i in range(len(words)-7): g = ' '.join(words[i:i+8]); ngc[g] = ngc.get(g,0)+1
            for g, c in ngc.items():
                if c >= 3:
                    f = answer.find(g); s2 = answer.find(g, f+len(g))
                    if s2 > 0: answer = answer[:s2].strip()
                    break
        en = set(m.group() for m in _NUMBER_RE.finditer(evidence))
        es = set(m.group().lower() for m in _STANDARD_RE.finditer(evidence))
        er = set(m.group().lower() for m in _SECTION_REF_RE.finditer(evidence))
        verified = []
        for sent in re.split(r'(?<=[.!?])\s+', answer.strip()):
            sn = set(m.group() for m in _NUMBER_RE.finditer(sent))
            ss = set(m.group().lower() for m in _STANDARD_RE.finditer(sent))
            sr = set(m.group().lower() for m in _SECTION_REF_RE.finditer(sent))
            if not sn and not ss and not sr: verified.append(sent); continue
            bad = False
            if sn and not sn & en: bad = True
            if ss:
                for st in ss:
                    if not any(st in e or e in st for e in es): bad = True; break
            if sr:
                for r in sr:
                    if r not in er and r not in evidence.lower(): bad = True; break
            if not bad: verified.append(sent)
        result = ' '.join(verified).strip()
        return result if result else "Not specified in the retrieved sections."


    def proof_quality_score(self, evidence, answer, family=None):
        """Score answer proof quality 0-100 based on evidence alignment."""
        if not evidence or not answer: return 0
        score = 0
        al = answer.lower()
        # Standards cited in answer that exist in evidence
        ans_stds = set(m.group().lower() for m in _STANDARD_RE.finditer(answer))
        ev_stds = set(m.group().lower() for m in _STANDARD_RE.finditer(evidence))
        if ans_stds:
            matched = sum(1 for s in ans_stds if any(s in e or e in s for e in ev_stds))
            score += min(30, matched * 15)
        else:
            score += 10  # no standards to verify
        # Requirement verbs present
        if _REQUIREMENT_WORDS_RE.search(answer): score += 15
        # Numbers in answer grounded in evidence
        ans_nums = set(m.group() for m in _NUMBER_RE.finditer(answer))
        ev_nums = set(m.group() for m in _NUMBER_RE.finditer(evidence))
        if ans_nums:
            grounded = sum(1 for n in ans_nums if n in ev_nums)
            score += min(25, int(grounded / len(ans_nums) * 25))
        else:
            score += 15
        # Not a generic abstention
        if "not specified" not in al: score += 15
        # Family-specific bonus
        if family == "metering" and any(w in al for w in ["monitoring", "monitor"]): score += 10
        elif family == "islanding" and any(w in al for w in ["islanded", "transition", "black start"]): score += 10
        elif family == "c5isr_power_grounding" and any(w in al for w in ["nfpa", "tia-607", "mil-std"]): score += 10
        elif family is None: score += 5
        return min(100, score)


if __name__ == "__main__":
    print("RelevanceFilterV4.5 loaded OK")
    for q, exp in [
        ("What are the metering and monitoring requirements?", "metering"),
        ("What are the islanding requirements for military microgrids?", "islanding"),
        ("What are the antenna and RF system requirements?", "antenna_rf"),
        ("What are the power and grounding requirements for C5ISR systems?", "c5isr_power_grounding"),
        ("What are the storage and closet space requirements?", "closet_storage"),
        ("What are the blast resistance requirements?", None),
    ]:
        f = detect_family(q)
        print(f"  [{str(f):25s}] {'OK' if f==exp else 'FAIL':5s} {q}")
