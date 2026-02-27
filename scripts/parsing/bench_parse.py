import time, numpy as np, torch
from exact.parser.loader import TrainedParser

motion = np.random.randn(200, 72).astype(np.float32)

# WITHOUT grammar constraint (faster)
print("Loading parser WITHOUT grammar...", flush=True)
parser_n = TrainedParser('results/parser/20260223_170723', use_grammar_constraint=False)

# Warmup
parser_n.parse(motion)

times_n = []
for i in range(5):
    t0 = time.time()
    r = parser_n.parse(motion)
    elapsed = time.time() - t0
    times_n.append(elapsed)
    print(f"  No-grammar run {i}: {elapsed:.3f}s -> {r[:80]}", flush=True)

del parser_n
torch.cuda.empty_cache()

# WITH grammar constraint 
print("\nLoading parser WITH grammar...", flush=True)
parser_g = TrainedParser('results/parser/20260223_170723', use_grammar_constraint=True)

# Warmup
parser_g.parse(motion)

times_g = []
for i in range(5):
    t0 = time.time()
    r = parser_g.parse(motion)
    elapsed = time.time() - t0
    times_g.append(elapsed)
    print(f"  Grammar run {i}: {elapsed:.3f}s -> {r[:80]}", flush=True)

avg_g = sum(times_g)/len(times_g)
avg_n = sum(times_n)/len(times_n)
print(f"\n{'='*60}", flush=True)
print(f"Average WITH grammar:    {avg_g:.3f}s/sample", flush=True)
print(f"Average WITHOUT grammar: {avg_n:.3f}s/sample", flush=True)
print(f"Speedup: {avg_g/avg_n:.1f}x", flush=True)
print(f"\nEstimate for 35837 ESK verb segments:", flush=True)
print(f"  WITH grammar:    {35837*avg_g/3600:.1f} hours", flush=True)
print(f"  WITHOUT grammar: {35837*avg_n/3600:.1f} hours", flush=True)
print(f"\nEstimate for ~107k total segments (3 datasets):", flush=True)
print(f"  WITH grammar:    {107511*avg_g/3600:.1f} hours", flush=True)
print(f"  WITHOUT grammar: {107511*avg_n/3600:.1f} hours", flush=True)
