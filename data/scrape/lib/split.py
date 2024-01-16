"""Split data into train, dev, and test."""

import os
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV_DIR = os.path.join(ROOT_DIR, "tsv")

def split(source: pd.Series, target: pd.Series) -> pd.Series:
    train_ratio = 0.80
    validation_ratio = 0.10
    test_ratio = 0.10

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(source, target, test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 10% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    return x_train, x_val, x_test, y_train, y_val, y_test

def find_file_in_subdirectories(directory, subdirectory):
  """
  Searches all subdirectories of a directory for a file and returns its full path.

  Args:
    directory: The directory to start searching from.
    filename: The name of the file to search for.

  Returns:
    The full path to the file if found, or None if not found.
  """

  for root, dirs, files in os.walk(directory):
    if subdirectory in dirs:
      # Found the file! Return its full path.
      return os.path.join(root, subdirectory)

  # File not found in any subdirectory.
  return None


def main(args):
    if args.dedupe == "y":
        with open(args.infile, "r", encoding="utf-8") as src:
                d = pd.read_csv(src, sep="\t")
                columns = ["source", "target"]
                d.columns = columns
                assert len(d.columns) == 2
                assert 'source' in d.columns
                assert 'target' in d.columns
                data = d.drop_duplicates(subset=["source"]).sample(n=3000, replace=False, random_state=42)
    else:
         with open(args.infile, "r", encoding="utf-8") as src:
                d = pd.read_csv(src, sep="\t").sample(n=3000, replace=False, random_state=42)
    # Assert that the dataframe contains two columns with the names "source" and "target"
    columns = ["source", "target"]
    data.columns = columns
    assert len(data.columns) == 2
    assert 'source' in data.columns
    assert 'target' in data.columns
    # Split Data Accordingly
    X = data.source
    y = data.target

    x_train, x_val, x_test, y_train, y_val, y_test = split(data["source"], data["target"])
    # Creating a train, dev, test set TSVs.
    train = {'source': x_train,
         'target': y_train}
    val = {'source': x_val,
         'target': y_val}
    test = {'source': x_test,
         'target': y_test}
    # Creating DataFrame by passing Dictionary
    train_set = pd.DataFrame(train)
    val_set = pd.DataFrame(val)
    test_set = pd.DataFrame(test)
    # Write sets to paths. 
    lang_path = find_file_in_subdirectories(TSV_DIR, args.language)
    train_path = f"{lang_path}/train/{args.language}_train.tsv"
    val_path = f"{lang_path}/val/{args.language}_val.tsv"
    test_path = f"{lang_path}/test/{args.language}_test.tsv"
    if not os.path.exists(train_path):
        train_set.to_csv(train_path, header=False, index=False, sep="\t")
    if not os.path.exists(val_path):
        val_set.to_csv(val_path, header=False, index=False, sep="\t")
    if not os.path.exists(test_path):
        test_set.to_csv(test_path, header=False, index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--infile", help="path to ingestion file for data splitting.")
    parser.add_argument("--language", help="string identifying the language being processed.")
    parser.add_argument("-dedupe", help="y/n indicating whether file needs deduping.")
    main(parser.parse_args())
