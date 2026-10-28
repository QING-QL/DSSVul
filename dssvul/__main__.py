import argparse
import json
from .train import run


def main():
    parser = argparse.ArgumentParser(description="DSSVul training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=["full", "nce", "ndp", "nbl", "nnrce"])
    args = parser.parse_args()
    with open(args.config) as stream:
        config = json.load(stream)
    for key in ("output", "seed", "variant"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    run(config)


if __name__ == "__main__":
    main()
