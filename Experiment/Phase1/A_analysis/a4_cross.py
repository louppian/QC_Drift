"""Collect existing checkpoint predictions into Result/Phase1/A_analysis/a4_cross.csv."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Experiment.Phase1.A_analysis import common as A


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    A.aggregate_stage("a4_cross")


if __name__ == "__main__":
    main()
