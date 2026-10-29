"""Write a sample CSV dataset: python scripts/make_sample_data.py --out data/sample"""
import _bootstrap  # noqa: F401
import argparse

from energy_detective.data.loaders import save_dataset
from energy_detective.data.simulator import generate_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/sample")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
save_dataset(generate_dataset(a.seed), a.out)
print("wrote", a.out)
