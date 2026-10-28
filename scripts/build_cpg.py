import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dssvul.datasets import validate_dataset


def main():
    parser = argparse.ArgumentParser(description="Build CPG inputs from canonical functions and fixed partitions")
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--joern", required=True)
    parser.add_argument("--extension", choices=["c", "cpp"], default="cpp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--java-options", default="-Xmx4g -XX:ActiveProcessorCount=4")
    args = parser.parse_args()
    verification = validate_dataset(args.data, args.splits, args.dataset)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "src").mkdir()
    csv.field_size_limit(64 * 1024 * 1024)
    sources = []
    with Path(args.data).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            filename = row["sample_id"] + "." + args.extension
            (out / "src" / filename).write_bytes(row["code"].encode("utf-8"))
            sources.append(dict(id=row["sample_id"], file=filename, label=int(row["target"])))
    (out / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    (out / "build_config.json").write_text(json.dumps({**vars(args), "input_verification": verification}, indent=2), encoding="utf-8")
    env = {**os.environ, "_JAVA_OPTIONS": args.java_options}
    joern = Path(args.joern).resolve()
    scripts = Path(__file__).resolve().parent
    with (out / "parse.log").open("w", encoding="utf-8") as log:
        subprocess.run([str(joern / "joern-parse"), str(out / "src"), "--language", "c",
                        "--output", str(out / "cpg.bin")], cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / "export.log").open("w", encoding="utf-8") as log:
        subprocess.run([str(joern / "joern"), "--script", str(scripts / "export_cpg.sc"),
                        "--param", "cpgFile=" + str(out / "cpg.bin"), "--param", "outFile=" + str(out / "graphs.json")],
                       cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(scripts / "prepare_cpg.py"), "--root", str(out),
                    "--output", str(out / "prepared"), "--splits", str(Path(args.splits).resolve()),
                    "--seed", str(args.seed)], check=True)


if __name__ == "__main__":
    main()
