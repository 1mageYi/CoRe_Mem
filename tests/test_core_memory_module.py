from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.memory import MemoryObservation, ResidualMemoryManager, VanillaCoreMemory


def test_residual_manager_keeps_most_salient_slots():
    memory = VanillaCoreMemory()
    memory.writer = memory.writer.__class__(
        updater=memory.writer.updater,
        residual_manager=ResidualMemoryManager(max_slots=2),
        residual_salience_floor=0.0,
    )
    memory.observe(MemoryObservation(latent=[1.0], timestamp="2026-04-04T00:00:00Z", salience=0.2, support=["a"]))
    memory.observe(MemoryObservation(latent=[2.0], timestamp="2026-04-04T00:01:00Z", salience=0.9, support=["b"]))
    state = memory.observe(
        MemoryObservation(latent=[3.0], timestamp="2026-04-04T00:02:00Z", salience=0.5, support=["c"])
    )

    assert state.residual_count() == 2
    assert [slot.support[0] for slot in state.residual_slots] == ["b", "c"]


def test_core_memory_updates_vector_and_returns_ranked_support():
    memory = VanillaCoreMemory()
    memory.observe(MemoryObservation(latent=[1.0, 0.0], timestamp="2026-04-04T00:00:00Z", salience=0.4, support=["s1"]))
    state = memory.observe(
        MemoryObservation(latent=[0.0, 1.0], timestamp="2026-04-04T00:01:00Z", salience=0.8, support=["s2"])
    )

    assert state.core_vector == [0.25, 0.5]

    result = memory.query([0.0, 1.0])
    assert result.supporting_slots[0].support == ["s2"]
    assert result.core_vector == [0.25, 0.5]
