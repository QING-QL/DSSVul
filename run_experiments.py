import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dssvul.train import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--variants", nargs="+", choices=["full", "nce", "ndp", "nbl", "nnrce"], default=["full"])
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    base = json.loads(Path(args.config).read_text())
    rows = []
    for seed in args.seeds:
        for variant in args.variants:
            output = Path(args.output_root) / f"{variant}_seed{seed}"
            config = {**base, "seed": seed, "variant": variant, "output": str(output)}
            if args.plan_only:
                print(json.dumps(config))
                continue
            result = run(config)
            rows.append(dict(seed=seed, variant=variant, **result["test"]))
            with (Path(args.output_root) / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    if rows:
        aggregate = {}
        for variant in args.variants:
            subset = [r for r in rows if r["variant"] == variant]
            aggregate[variant] = {m: {"mean": float(np.mean([r[m] for r in subset])),
                                       "std": float(np.std([r[m] for r in subset], ddof=1)) if len(subset)>1 else None}
                                  for m in ("precision", "recall", "f1", "mcc", "auprc")}
        (Path(args.output_root) / "aggregate.json").write_text(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
