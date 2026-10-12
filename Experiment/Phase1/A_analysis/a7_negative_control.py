"""Collect existing checkpoint predictions into Result/Phase1/A_analysis/a7_negative_control.csv."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Experiment.Phase1.A_analysis import common as A


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    A.aggregate_stage("a7_negative_control")


if __name__ == "__main__":
    main()
