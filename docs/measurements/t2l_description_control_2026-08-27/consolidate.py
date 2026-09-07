"""Aligned vs junk descriptions on the released T2L SFT checkpoint."""
import re, sys, statistics as st

ZS = {"arc_challenge":65.70,"arc_easy":77.48,"boolq":71.56,"hellaswag":49.67,
      "openbookqa":55.00,"piqa":73.01,"winogrande":45.54,"gsm8k":40.71,"humaneval":37.80}

def parse(path, n):
    out, cur = {}, None
    for line in open(path):
        line = line.strip()
        m = re.match(r"^=== (\w+) ===$", line)
        if m: cur = m.group(1); out[cur] = []; continue
        m = re.search(r"'(?:acc|humaneval_base_pass@1)':\s*(?:np\.float64\()?([0-9.]+)", line)
        if m and cur: out[cur].append(100*float(m.group(1)))
    return {k: v for k, v in out.items() if len(v) == 2*n}

rows = {}
rows.update(parse(sys.argv[1], 2))   # exp1: 2 descs
rows.update(parse(sys.argv[2], 3))   # exp3: 3 descs

print(f"{'task':15s} {'aligned':>8s} {'junk':>8s} {'gap':>7s} {'噪声':>6s} "
      f"{'zero-shot':>9s} {'T2L增益':>8s} {'描述占比':>8s}")
shares, gaps = [], []
for t, v in sorted(rows.items()):
    n = len(v)//2
    a, j = v[:n], v[n:]
    am, jm = sum(a)/n, sum(j)/n
    noise = max(max(a)-min(a), max(j)-min(j))
    gain = am - ZS[t]
    gap = am - jm
    gaps.append(gap)
    share = 100*gap/gain if gain > 1 else float("nan")
    if gain > 1: shares.append(share)
    ss = f"{share:.0f}%" if gain > 1 else "n/a"
    print(f"{t:15s} {am:8.2f} {jm:8.2f} {gap:+7.2f} {noise:6.2f} "
          f"{ZS[t]:9.2f} {gain:+8.2f} {ss:>8s}")
print(f"\n任务数 {len(rows)}；gap 中位 {st.median(gaps):.2f}；"
      f"描述占比 中位 {st.median(shares):.0f}% 均值 {sum(shares)/len(shares):.0f}%")
