"""Quick analysis of gold rank distribution in a run's debug_samples.jsonl"""
import json, pathlib, statistics, sys

run_dir = pathlib.Path("outputs/locomo_eval/20260427_170019")
debug_path = run_dir / "debug_samples.jsonl"

ranks = []
field_names_printed = False

for line in debug_path.open(encoding="utf-8"):
    s = json.loads(line)
    diag = s.get("retrieval_diagnostics", {})

    if not field_names_printed:
        print("retrieval_diagnostics 字段:", list(diag.keys()))
        field_names_printed = True

    sem_rank = diag.get("best_semantic_rank_global")   # rank in full expanded pool by semantic sim
    rerank_pos = diag.get("gold_rerank_position_among_expanded")  # rank after fusion re-ranking
    if sem_rank is not None:
        ranks.append((int(sem_rank), int(rerank_pos) if rerank_pos is not None else 9999))

if not ranks:
    print("\nNo rank field found. Printing first sample's full retrieval_diagnostics:")
    debug_path2 = run_dir / "debug_samples.jsonl"
    sample = json.loads(debug_path2.open(encoding="utf-8").readline())
    print(json.dumps(sample.get("retrieval_diagnostics", {}), indent=2, ensure_ascii=False))
    sys.exit(0)

sem_ranks = [r[0] for r in ranks]
rerank_ranks = [r[1] for r in ranks if r[1] != 9999]
n = len(ranks)

def pct(lst, p):
    idx = max(0, int(len(lst)*p) - 1)
    return sorted(lst)[idx]

print(f"\n=== 语义 rank (best_semantic_rank_global) — 扩展候选池内的纯语义排名 ===")
print(f"样本数: {n}, 有效: {sum(1 for r in sem_ranks if r < 9999)}")
valid_sem = sorted(r for r in sem_ranks if r < 9999)
print(f"平均={statistics.mean(valid_sem):.1f}, 中位数={statistics.median(valid_sem):.1f}")
for cutoff in [8, 12, 16, 20, 24, 32, 40]:
    covered = sum(1 for r in valid_sem if r <= cutoff)
    print(f"  k={cutoff:2d}: 覆盖 {covered}/{len(valid_sem)} = {covered/len(valid_sem):.1%}")

print(f"\n=== 融合 rerank 后 rank (gold_rerank_position_among_expanded) ===")
valid_rr = sorted(rerank_ranks)
if valid_rr:
    print(f"样本数: {len(valid_rr)}")
    print(f"平均={statistics.mean(valid_rr):.1f}, 中位数={statistics.median(valid_rr):.1f}")
    for cutoff in [8, 12, 16, 20, 24, 32, 40]:
        covered = sum(1 for r in valid_rr if r <= cutoff)
        print(f"  k={cutoff:2d}: 覆盖 {covered}/{len(valid_rr)} = {covered/len(valid_rr):.1%}")
