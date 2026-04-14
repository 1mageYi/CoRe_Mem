"""
vec2text uses datasets.map(num_proc=get_num_proc()) where get_num_proc() ~= cpu_count.
On Linux, forked workers cannot use CUDA tensors after the parent initialized CUDA
("Cannot re-initialize CUDA in forked subprocess"). Hypothesis precompute runs the
inversion model on GPU inside map workers -> crash unless num_proc is 1.

Patch get_num_proc in every module that holds a direct import reference.
Controlled by env VEC2TEXT_MAP_NUM_PROC (default 1).
"""
from __future__ import annotations

import os


def apply_vec2text_map_num_proc_patch() -> None:
    import vec2text.data_helpers as vdata
    import vec2text.experiments as vexp
    import vec2text.utils.utils as vuu

    def _map_num_proc() -> int:
        return max(1, int(os.environ.get("VEC2TEXT_MAP_NUM_PROC", "1")))

    vuu.get_num_proc = _map_num_proc  # type: ignore[assignment]
    vexp.get_num_proc = _map_num_proc  # type: ignore[assignment]
    vdata.get_num_proc = _map_num_proc  # type: ignore[assignment]
