#!/usr/bin/env python3
"""Relevance-based chunk filter using sentence embeddings."""

import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer


class RelevanceFilter:
    """Embed text chunks and filter by cosine similarity to a query."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._original_count = 0
        self._kept_count = 0

    def chunk(self, text, chunk_size=3):
        """Split text into groups of N sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s.strip()]
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size])
            chunks.append(chunk)
        self._original_count = len(chunks)
        return chunks

    def filter(self, chunks, query, top_k=3):
        """Return the top_k chunks most similar to query, preserving order."""
        all_texts = chunks + [query]
        embeddings = self.model.encode(all_texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)

        chunk_embs = embeddings[:-1]
        query_emb = embeddings[-1]

        # Cosine similarity
        norms = np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb)
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = np.dot(chunk_embs, query_emb) / norms

        # Get top_k indices, then sort by original position
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_indices_sorted = sorted(top_indices)

        self._kept_count = len(top_indices_sorted)
        self._similarities = similarities

        return top_indices_sorted, similarities

    def stats(self):
        """Return filtering statistics."""
        ratio = self._kept_count / self._original_count if self._original_count else 0
        return {
            "original_chunks": self._original_count,
            "kept_chunks": self._kept_count,
            "reduction_ratio": ratio,
        }
    def precompute(self, chunks):
        """Pre-encode chunks. Call once at startup."""
        self._cached_chunks = chunks
        self._cached_embs = self.model.encode(chunks, convert_to_numpy=True)

    def filter_cached(self, query, top_k=10):
        """Filter using pre-computed embeddings. Fast for real-time use."""
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        norms = np.linalg.norm(self._cached_embs, axis=1) * np.linalg.norm(query_emb)
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = np.dot(self._cached_embs, query_emb) / norms
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_indices_sorted = sorted(top_indices)
        return [self._cached_chunks[i] for i in top_indices_sorted], similarities



# ---------------------------------------------------------------------------
# Test: 50-sentence construction spec
# ---------------------------------------------------------------------------
SPEC_DOC = (
    # Site Work (sentences 1-8)
    "The site contractor shall complete all earthwork and grading operations prior to foundation work. "
    "Topsoil shall be stripped to a depth of 12 inches and stockpiled on site for later use. "
    "Erosion control measures including silt fencing and sediment basins shall be installed per the SWPPP. "
    "The geotechnical engineer has approved the bearing capacity at 3000 PSF for spread footings. "
    "All underground utilities shall be located and marked before excavation begins. "
    "Temporary construction roads shall use 6 inches of compacted gravel base. "
    "Dewatering operations are required in the northwest corner due to high water table conditions. "
    "The site lighting plan includes 20-foot pole-mounted LED fixtures at 50-foot spacing along access roads. "

    # Structural (sentences 9-16)
    "Structural steel erection shall proceed from grid A to grid J in sequence. "
    "All moment connections require CJP welds with ultrasonic testing per AWS D1.1. "
    "The precast concrete panels for the building envelope weigh approximately 8 tons each. "
    "Anchor bolt placement tolerances shall not exceed 1/8 inch from plan location. "
    "The raised access floor system requires a minimum 18-inch plenum depth. "
    "Seismic bracing for all equipment over 400 pounds shall be designed per ASCE 7. "
    "The roof structure consists of open web steel joists at 5-foot spacing with metal deck. "
    "Fireproofing of structural steel shall achieve a 2-hour fire rating per UL assembly. "

    # HVAC (sentences 17-26)
    "The HVAC system design includes four rooftop units serving the office areas. "
    "Each rooftop unit is rated at 25 tons cooling capacity with gas heat. "
    "Variable air volume boxes with DDC controls serve all interior zones. "
    "The commissioning agent shall verify proper airflow at each VAV terminal. "
    "HVAC commissioning requirements include functional performance testing of all air handling units. "
    "The building automation system shall demonstrate trending and alarm capabilities during commissioning. "
    "Ductwork leakage testing shall not exceed 4 CFM per 100 square feet of duct surface area. "
    "Refrigerant piping shall be pressure tested at 350 PSI for 24 hours with nitrogen. "
    "The energy recovery ventilator shall achieve a minimum 72 percent effectiveness rating. "
    "HVAC controls sequences of operation shall be reviewed and approved prior to commissioning startup. "

    # Electrical (sentences 27-36)
    "The main electrical service is 4000 amps at 480/277 volts three-phase. "
    "Two 2000 KVA dry-type transformers feed the main distribution switchgear. "
    "The emergency generator is rated at 1500 KW with a 500-gallon belly tank. "
    "Automatic transfer switches shall be tested monthly with load bank verification annually. "
    "The lighting control system uses DALI protocol with daylight harvesting in perimeter zones. "
    "All branch circuit wiring in data center areas shall be installed in cable tray. "
    "The grounding system includes a ground ring with 20-foot driven rods at building corners. "
    "Arc flash labels shall be applied to all panels and switchgear per NFPA 70E. "
    "The electrical contractor shall provide coordination study results before energization. "
    "Temporary power distribution shall maintain 200 amps per floor during construction. "

    # Plumbing (sentences 37-42)
    "Domestic water service enters the building through a 4-inch copper main with backflow preventer. "
    "The hot water system uses two 100-gallon commercial water heaters in parallel configuration. "
    "Sanitary sewer connects to the municipal system via a 6-inch PVC lateral at the property line. "
    "Roof drainage includes internal conductors sized for a 100-year storm event. "
    "All plumbing fixtures shall be low-flow models meeting WaterSense certification requirements. "
    "The grease interceptor for the kitchen area is sized at 50 GPM with 100 pounds capacity. "

    # Fire Protection (sentences 43-50)
    "The fire suppression system is a wet pipe sprinkler design per NFPA 13. "
    "Sprinkler coverage in the data center uses a pre-action dry pipe system with double interlock. "
    "The fire alarm control panel is an addressable system with voice evacuation capability. "
    "Smoke detectors are required above and below the raised access floor in server rooms. "
    "Fire dampers shall be installed at all rated wall and floor penetrations. "
    "The fire pump is a 750 GPM electric-driven unit with jockey pump and controller. "
    "Standpipe connections are required at each stairwell landing per the fire code. "
    "The clean agent suppression system in the MDF uses FM-200 with a 10-second discharge time."
)

QUERY = "What are the HVAC commissioning requirements?"
def main():
    print("=" * 70)
    print("  Relevance Filter Test")
    print("=" * 70)

    print("\nLoading embedding model...")
    t0 = time.perf_counter()
    rf = RelevanceFilter()
    print("Model loaded in {:.1f}s".format(time.perf_counter() - t0))

    # Chunk the document
    chunks = rf.chunk(SPEC_DOC, chunk_size=3)
    print("\nDocument: {} sentences -> {} chunks (3 sentences each)".format(
        len(re.split(r'(?<=[.!?])\s+', SPEC_DOC.strip())), len(chunks)))
    print("Query: \"{}\"".format(QUERY))

    # Filter
    t0 = time.perf_counter()
    kept_indices, similarities = rf.filter(chunks, QUERY, top_k=3)
    filter_time = time.perf_counter() - t0
    print("Filter time: {:.2f}s".format(filter_time))

    stats = rf.stats()
    print("\n" + "-" * 70)
    print("  ALL CHUNKS WITH SIMILARITY SCORES")
    print("-" * 70)

    for i, chunk in enumerate(chunks):
        marker = " <<<< KEPT" if i in kept_indices else ""
        print("\n  Chunk {:2d}  [sim: {:.4f}]{}".format(i, similarities[i], marker))
        # Wrap long text
        words = chunk.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 78:
                print(line)
                line = "    " + w
            else:
                line = line + " " + w if line.strip() else "    " + w
        if line.strip():
            print(line)

    print("\n" + "=" * 70)
    print("  FILTERING RESULTS")
    print("=" * 70)
    print("  Original chunks : {}".format(stats["original_chunks"]))
    print("  Kept chunks     : {}".format(stats["kept_chunks"]))
    print("  Reduction ratio  : {:.1%} kept ({:.1%} removed)".format(
        stats["reduction_ratio"], 1 - stats["reduction_ratio"]))

    print("\n" + "-" * 70)
    print("  FILTERED TEXT (chunks {})".format(list(kept_indices)))
    print("-" * 70)
    filtered = " ".join(chunks[i] for i in kept_indices)
    print()
    print(filtered)
    print()
    print("  Original chars: {}  ->  Filtered chars: {}".format(
        len(SPEC_DOC), len(filtered)))
    print("=" * 70)


if __name__ == "__main__":
    main()
