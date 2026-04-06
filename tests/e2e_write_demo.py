"""End-to-end demo: write 50 user facts and observe core/residual dynamics."""

from __future__ import annotations

import sys
import time

import numpy as np

from core_mem.config import MemorySettings
from core_mem.writer import MemoryWriter
from core_mem.slot import cosine_similarity


FACTS = [
    # --- Food preferences (cluster 1) ---
    "I love Italian food, especially pasta and pizza.",
    "My favorite cuisine is Italian, I eat it almost every week.",
    "I tried a new Italian restaurant last night, the carbonara was amazing.",
    "I prefer thin-crust pizza over deep-dish.",
    "I'm thinking about learning to make homemade pasta.",
    "Italian wine pairs perfectly with my dinner.",
    "I don't like spicy food at all.",
    "I recently started eating more salads for lunch.",
    # --- Location / living (cluster 2) ---
    "I live in New York City, in the Upper West Side.",
    "I've been living in NYC for about five years now.",
    "My apartment is a small one-bedroom but I love the neighborhood.",
    "I'm considering moving to San Francisco next year.",
    "The rent in NYC is getting too expensive for me.",
    "I grew up in a small town in Ohio before moving to the city.",
    "I miss the quiet life sometimes but I love the energy of NYC.",
    # --- Work / career (cluster 3) ---
    "I work as a software engineer at a startup.",
    "My team is building a machine learning platform.",
    "I've been coding in Python for about eight years.",
    "I'm interested in transitioning to a research role.",
    "My company just got Series B funding.",
    "I usually work from home three days a week.",
    "I'm thinking about getting a master's degree in AI.",
    "I have a job interview at Google next month.",
    # --- Hobbies (cluster 4) ---
    "I play guitar in my free time, mostly acoustic.",
    "I've been playing guitar for about ten years.",
    "I recently started learning jazz guitar.",
    "I also enjoy hiking on weekends when the weather is nice.",
    "I ran my first half marathon last spring.",
    "I'm training for a full marathon next year.",
    "I like reading science fiction novels before bed.",
    "My favorite author is Isaac Asimov.",
    # --- Pets (cluster 5) ---
    "I have a golden retriever named Max.",
    "Max is three years old and very energetic.",
    "I take Max to the dog park every morning.",
    "I'm thinking about getting a second dog, maybe a beagle.",
    "Max loves playing fetch in Central Park.",
    # --- Travel (cluster 6) ---
    "I'm planning a trip to Japan next spring.",
    "I've always wanted to visit Kyoto and see the temples.",
    "I visited Paris last summer and loved it.",
    "I want to go back to Europe, maybe Spain next time.",
    "I prefer traveling solo, it gives me more freedom.",
    # --- Health (cluster 7) ---
    "I've been trying to sleep earlier, aiming for 11 PM.",
    "I started meditating every morning for ten minutes.",
    "I cut down on coffee, now I only drink one cup a day.",
    "I'm a bit lactose intolerant so I avoid dairy when I can.",
    # --- Miscellaneous ---
    "My birthday is on March 15th.",
    "I'm turning 29 this year.",
    "I speak English and a little bit of Japanese.",
    "I prefer dark mode on all my devices.",
    "My phone is an iPhone 15 Pro.",
]


def run_demo() -> None:
    cfg = MemorySettings(
        core_slots=32,
        residual_slots=64,
        merge_threshold=0.5,
        core_threshold=0.5,
        promotion_merge_count=5,
        stability_window=3,
        stability_epsilon=0.05,
        recency_weight=1.5,
    )
    writer = MemoryWriter(settings=cfg)

    print("=" * 70)
    print("CoRe Memory Write Path — End-to-End Demo (50 facts)")
    print("=" * 70)
    print(f"Config: core_slots={cfg.core_slots}, residual_slots={cfg.residual_slots}, "
          f"merge_threshold={cfg.merge_threshold}, promotion_merge_count={cfg.promotion_merge_count}")
    print()

    # Load embedding model once (will print download progress on first run)
    print("Loading embedding model (first call may download ~80MB)...")
    from core_mem.embedding import encode
    _ = encode("warmup")
    print("Model loaded.\n")

    promotions = []

    for i, fact in enumerate(FACTS, 1):
        result = writer.write(fact, provenance=f"demo_t{i}")

        status = []
        if result.merged:
            status.append("MERGED")
        else:
            status.append("NEW")
        if result.promoted:
            status.append("-> PROMOTED TO CORE")
            promotions.append((i, fact))

        print(f"[{i:2d}/{len(FACTS)}] {' | '.join(status):30s} | "
              f"R={writer.residual.count:2d}  C={writer.core.count:2d} | "
              f"mc={result.slot.merge_count:2d} | {fact[:60]}")

    # --- Summary ---
    print()
    print("=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    print(f"Residual slots used: {writer.residual.count} / {cfg.residual_slots}")
    print(f"Core slots used:     {writer.core.count} / {cfg.core_slots}")
    print()

    if promotions:
        print(f"Promotions triggered: {len(promotions)}")
        for idx, fact in promotions:
            print(f"  Fact #{idx}: {fact[:70]}")
    else:
        print("No promotions triggered (expected with only 50 diverse facts).")
    print()

    # --- Residual slot overview ---
    print("--- Residual Slots ---")
    for j, slot in enumerate(writer.residual.slots):
        # Find the original fact closest to this slot's embedding
        fact_sims = [cosine_similarity(slot.embedding, encode(f)) for f in FACTS]
        best_fact_idx = int(np.argmax(fact_sims))
        print(f"  Slot {j:2d}: merge_count={slot.merge_count:2d}, "
              f"sim={fact_sims[best_fact_idx]:.3f}, "
              f"closest_fact=\"{FACTS[best_fact_idx][:55]}...\"")

    if writer.core.count > 0:
        print()
        print("--- Core Slots ---")
        for j, slot in enumerate(writer.core.slots):
            fact_sims = [cosine_similarity(slot.embedding, encode(f)) for f in FACTS]
            best_fact_idx = int(np.argmax(fact_sims))
            print(f"  Slot {j:2d}: merge_count={slot.merge_count:2d}, "
                  f"sim={fact_sims[best_fact_idx]:.3f}, "
                  f"closest_fact=\"{FACTS[best_fact_idx][:55]}...\"")

    # --- Persistence test ---
    print()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        writer.save(tmpdir)
        writer2 = MemoryWriter(settings=cfg)
        writer2.load(tmpdir)
        print(f"Persistence test: saved & reloaded "
              f"(R={writer2.residual.count}, C={writer2.core.count}) ✓")

    print()
    print("Demo complete.")


if __name__ == "__main__":
    run_demo()
