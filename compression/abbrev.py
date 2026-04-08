#!/usr/bin/env python3
"""Construction Domain Semantic Abbreviation Compressor.

Compresses construction industry text using a dictionary of ~120 standard
abbreviations spanning CSI divisions, MEP systems, documents, data center
infrastructure, scheduling, and general construction terminology.
"""

import re
from collections import defaultdict


class ConstructionCompressor:
    """Bidirectional compressor for construction domain text."""

    def __init__(self):
        self._build_dictionary()
        self._stats = defaultdict(int)
        self._original_len = 0
        self._compressed_len = 0

    def _build_dictionary(self):
        """~120 construction-domain term -> abbreviation mappings."""
        self._terms = {
            # CSI Divisions
            "general conditions":       "GC/GEN",
            "site construction":        "SITCON",
            "concrete":                 "CONC",
            "masonry":                  "MSNRY",
            "structural steel":         "STR-STL",
            "rough carpentry":          "RGH-CARP",
            "finish carpentry":         "FIN-CARP",
            "waterproofing":            "WP",
            "insulation":               "INSUL",
            "roofing":                  "RFG",
            "sheet metal":              "SHT-MTL",
            "doors and hardware":       "DR-HW",
            "glazing":                  "GLZ",
            "drywall":                  "DW",
            "flooring":                 "FLR",
            "painting":                 "PTG",
            "specialties":              "SPEC",
            "equipment":                "EQUIP",
            "furnishings":              "FURN",
            "conveying systems":        "CONV-SYS",
            "fire suppression":         "F-SUPP",
            "fire protection":          "FP",
            "plumbing":                 "PLMB",
            "mechanical":               "MECH",
            "electrical":               "ELEC",
            "earthwork":                "ERTHWK",
            "demolition":               "DEMO",
            "reinforcing steel":        "REBAR",

            # MEP
            "heating ventilation and air conditioning": "HVAC",
            "air handling unit":        "AHU",
            "variable air volume":      "VAV",
            "rooftop unit":             "RTU",
            "chilled water":            "CHW",
            "hot water":                "HW",
            "condenser water":          "CW",
            "building automation system": "BAS",
            "building management system": "BMS",
            "energy management system": "EMS",
            "direct digital control":   "DDC",
            "programmable logic controller": "PLC",
            "motor control center":     "MCC",
            "variable frequency drive": "VFD",
            "circuit breaker":          "CB",
            "distribution panel":       "DP",
            "switchgear":               "SWGR",
            "transformer":              "XFMR",
            "uninterruptible power supply": "UPS",
            "emergency power":          "EMER-PWR",
            "generator":                "GEN",
            "automatic transfer switch": "ATS",
            "fire alarm":               "FA",
            "fire alarm control panel": "FACP",
            "sprinkler":                "SPRK",
            "smoke detector":           "SD",
            "backflow preventer":       "BFP",
            "domestic water":           "DW-P",
            "sanitary sewer":           "SAN",
            "storm drain":              "STM-DR",
            "grease interceptor":       "GI",

            # Documents
            "request for information":  "RFI",
            "request for proposal":     "RFP",
            "request for quotation":    "RFQ",
            "change order":             "CO",
            "change order request":     "COR",
            "construction change directive": "CCD",
            "submittal":                "SUBM",
            "submittals":               "SUBMs",
            "shop drawings":            "SD-DWG",
            "as-built drawings":        "ABD",
            "record drawings":          "REC-DWG",
            "specifications":           "SPECS",
            "punch list":               "PL",
            "daily report":             "DR",
            "safety data sheet":        "SDS",
            "operation and maintenance": "O&M",
            "certificate of occupancy": "C-of-O",
            "notice to proceed":        "NTP",
            "substantial completion":   "SC",
            "final completion":         "FC",
            "letter of intent":         "LOI",
            "guaranteed maximum price": "GMP",
            "bid documents":            "BID-DOC",

            # Data Center
            "data center":              "DC",
            "computer room air conditioning": "CRAC",
            "computer room air handler": "CRAH",
            "raised access floor":      "RAF",
            "hot aisle containment":    "HAC",
            "cold aisle containment":   "CAC",
            "power distribution unit":  "PDU",
            "remote power panel":       "RPP",
            "static transfer switch":   "STS",
            "battery monitoring system": "BMON",
            "rack unit":                "RU",
            "structured cabling":       "SC-CBL",
            "fiber optic":              "FO",
            "cable tray":               "CT",
            "environmental monitoring": "EMON",
            "redundancy":               "REDUN",

            # Schedule
            "critical path method":     "CPM",
            "work breakdown structure": "WBS",
            "milestone":                "MS",
            "predecessor":              "PRED",
            "successor":                "SUCC",
            "float":                    "FLT",
            "baseline schedule":        "BL-SCHED",
            "look-ahead schedule":      "LA-SCHED",
            "schedule of values":       "SOV",
            "percent complete":         "PCT-COMP",
            "earned value":             "EV",
            "planned value":            "PV",
            "schedule performance index": "SPI",
            "cost performance index":   "CPI",
            "liquidated damages":       "LD",

            # General Construction
            "general contractor":       "GC",
            "subcontractor":            "SUB",
            "construction manager":     "CM",
            "construction documents":   "CD",
            "design development":       "DD",
            "schematic design":         "SD-PH",
            "owner's representative":   "OR",
            "architect of record":      "AOR",
            "engineer of record":       "EOR",
            "quality control":          "QC",
            "quality assurance":        "QA",
            "inspection":               "INSP",
            "commissioning":            "CX",
            "retro-commissioning":      "RCX",
            "building envelope":        "BENV",
            "curtain wall":             "CW-WALL",
            "precast concrete":         "PC-CONC",
            "cast in place":            "CIP",
            "post-tensioned":           "PT",
            "means and methods":        "M&M",
            "temporary construction":   "TEMP-CON",
            "safety":                   "SFTY",
            "occupational safety and health administration": "OSHA",
            "personal protective equipment": "PPE",
            "job hazard analysis":      "JHA",
            "total project cost":       "TPC",
        }

        # Reverse map (abbrev -> full term)
        self._reverse = {v: k for k, v in self._terms.items()}

        # Sort terms longest-first so longer phrases match before substrings
        self._sorted_terms = sorted(self._terms.keys(), key=len, reverse=True)

        # Pre-compile patterns (case-insensitive, whole-word)
        self._patterns = []
        for term in self._sorted_terms:
            pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            self._patterns.append((pat, self._terms[term], term))

    def compress(self, text):
        """Replace known construction terms with abbreviations."""
        self._stats.clear()
        self._original_len = len(text)
        result = text

        for pat, abbrev, term in self._patterns:
            new_result, count = pat.subn(abbrev, result)
            if count:
                self._stats[term] += count
                result = new_result

        self._compressed_len = len(result)
        return result

    def decompress(self, text):
        """Expand abbreviations back to full terms."""
        result = text
        sorted_abbrevs = sorted(self._reverse.keys(), key=len, reverse=True)
        for abbrev in sorted_abbrevs:
            full = self._reverse[abbrev]
            pat = re.compile(r"\b" + re.escape(abbrev) + r"\b")
            result = pat.sub(full, result)
        return result

    def stats_report(self):
        """Return a formatted compression statistics report."""
        if not self._original_len:
            return "No compression performed yet."

        saved = self._original_len - self._compressed_len
        ratio = (saved / self._original_len) * 100 if self._original_len else 0

        lines = []
        lines.append("=" * 60)
        lines.append("  CONSTRUCTION ABBREVIATION COMPRESSION REPORT")
        lines.append("=" * 60)
        lines.append(f"  Original length  : {self._original_len:,} chars")
        lines.append(f"  Compressed length: {self._compressed_len:,} chars")
        lines.append(f"  Saved            : {saved:,} chars ({ratio:.1f}%)")
        lines.append(f"  Dictionary size  : {len(self._terms)} terms")
        lines.append(f"  Terms matched    : {len(self._stats)}")
        lines.append(f"  Total replacements: {sum(self._stats.values())}")
        lines.append("-" * 60)
        lines.append("  SUBSTITUTIONS:")
        lines.append("-" * 60)

        for term, count in sorted(self._stats.items(), key=lambda x: -x[1]):
            abbrev = self._terms[term]
            lines.append(f"    {term:<45} -> {abbrev:<10} x{count}")

        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    sample = (
        "The general contractor is coordinating the heating ventilation "
        "and air conditioning installation across all three floors. The "
        "subcontractor submitted a request for information regarding the "
        "variable air volume box specifications and is awaiting approval "
        "on the submittals for the air handling unit and variable frequency "
        "drive components. Meanwhile, the construction manager has flagged "
        "that the data center computer room air conditioning units and the "
        "uninterruptible power supply installation are behind the baseline "
        "schedule by two weeks. The power distribution unit and remote power "
        "panel work is on the critical path method schedule and any delay "
        "will affect substantial completion. The commissioning agent is "
        "reviewing the building automation system integration with the fire "
        "alarm control panel and fire protection sprinkler system. Quality "
        "control inspections on the structural steel connections and precast "
        "concrete panels are scheduled for next week. The architect of record "
        "issued a change order for additional fire suppression coverage in "
        "the raised access floor area. All shop drawings for the switchgear "
        "and transformer installation must be resubmitted before the notice "
        "to proceed is granted. The owner's representative requested an "
        "updated schedule of values and the look-ahead schedule showing the "
        "punch list completion timeline. Personal protective equipment "
        "compliance and job hazard analysis documentation are required for "
        "all workers entering the mechanical and electrical zones."
    )

    comp = ConstructionCompressor()

    print("ORIGINAL TEXT:")
    print("-" * 60)
    print(sample)
    print()

    compressed = comp.compress(sample)

    print("COMPRESSED TEXT:")
    print("-" * 60)
    print(compressed)
    print()

    print(comp.stats_report())
    print()

    # Round-trip verification
    decompressed = comp.decompress(compressed)
    match = decompressed.lower().strip() == sample.lower().strip()
    print(f"Round-trip decompression match: {'PASS' if match else 'FAIL'}")
    if not match:
        print("\nDECOMPRESSED TEXT:")
        print(decompressed)
