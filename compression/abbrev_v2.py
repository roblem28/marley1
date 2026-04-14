"""
abbrev_v2.py - Token-optimized construction compressor for Qwen2.5 BPE.
Removes 20 terms that increase token count vs original.
Based on token audit against Qwen2.5-3B tokenizer April 13 2026.
"""
import re
from collections import defaultdict

WORSE_TERMS = {
    "sheet metal", "equipment", "fire suppression", "emergency power",
    "bid documents", "percent complete", "general conditions",
    "site construction", "structural steel", "rough carpentry",
    "earthwork", "storm drain", "shop drawings", "record drawings",
    "specifications", "baseline schedule", "inspection", "precast concrete",
    "temporary construction", "safety"
}

class ConstructionCompressor:
    def __init__(self):
        self._build_dictionary()
        self._stats = defaultdict(int)

    def _build_dictionary(self):
        from abbrev import ConstructionCompressor as Original
        orig = Original()
        self._terms = {k: v for k, v in orig._terms.items() if k not in WORSE_TERMS}
        self._pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in sorted(self._terms, key=len, reverse=True)) + r')\b',
            re.IGNORECASE
        )

    def compress(self, text):
        def replace(m):
            term = m.group(0).lower()
            return self._terms.get(term, m.group(0))
        return self._pattern.sub(replace, text)

    def decompress(self, text):
        from abbrev import ConstructionCompressor as Original
        orig = Original()
        rev = {v: k for k, v in self._terms.items()}
        pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(rev, key=len, reverse=True)) + r')\b')
        return pattern.sub(lambda m: rev.get(m.group(0), m.group(0)), text)
