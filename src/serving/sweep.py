"""Serving sweep: cartesian product of knobs, precision vs throughput per point."""
import argparse
import itertools

import yaml


def expand_grid(axes):
    if not axes:
        return [{}]
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(axes[k] for k in keys))]


def run_point(cfg, point):
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        axes = yaml.safe_load(f).get("sweep", {})
    print(f"{len(expand_grid(axes))} sweep points")
