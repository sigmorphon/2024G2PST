#!/usr/bin/env python3
"""Create orthographies datasets."""

import os
import argparse
import glob

import split
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV_DIR = os.path.join(ROOT_DIR, "tsv")

from typing import List

def join_datasets(glob_file_paths: List[str], outpath: str) -> pd.DataFrame:
    dfs = []
    for path in glob_file_paths:
        if path != outpath:
            df = pd.read_csv(path, sep="\t")
            columns = ["source", "target"]
            df.columns = columns
            assert len(df.columns) == 2
            assert 'source' in df.columns
            assert 'target' in df.columns
            dfs.append(df)
    # Concatenate Datasets
    data = pd.concat(dfs)
    data.sample(frac=1)
    return data

def main(args):
    # Find language set
    orth_path = split.find_file_in_subdirectories(TSV_DIR, args.orth)
    train_files = glob.glob(f'{orth_path}/**/train/*.tsv', recursive=True)
    val_files = glob.glob(f'{orth_path}/**/val/*.tsv', recursive=True)
    test_files = glob.glob(f'{orth_path}/**/test/*.tsv', recursive=True)
    # Create outpaths
    train_path = f"{orth_path}/{args.orth}/train/{args.orth}_train.tsv"
    val_path = f"{orth_path}/{args.orth}/val/{args.orth}_val.tsv"
    test_path = f"{orth_path}/{args.orth}/test/{args.orth}_test.tsv"
    # Merge Datasets.
    train_set = join_datasets(train_files, train_path)
    val_set = join_datasets(val_files, val_path)
    test_set = join_datasets(test_files, test_path)
    # Write sets.
    train_set.to_csv(train_path, header=False, index=False, sep="\t")
    val_set.to_csv(val_path, header=False, index=False, sep="\t")
    test_set.to_csv(test_path, header=False, index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-orth", help="string identifying the orthography being processed.")
    main(parser.parse_args())