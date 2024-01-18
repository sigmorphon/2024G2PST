#!/usr/bin/env python3
"""Filter language phones."""

import argparse
import os
import glob

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHONES_DIR = os.path.join(ROOT_DIR, "phones")
TSV_DIR = os.path.join(ROOT_DIR, "tsv")


def find_directory_in_subdirectories(directory, subdirectory):
  """
  Searches all subdirectories of a directory for a file and returns its full path.

  Args:
    directory: The directory to start searching from.
    filename: The name of the file to search for.

  Returns:
    The full path to the file if found, or None if not found.
  """

  for root, dirs, _ in os.walk(directory):
    if subdirectory in dirs:
      # Found the file! Return its full path.
      return os.path.join(root, subdirectory)

  # File not found in any subdirectory.
  return None


def _create_set(data):
    phones_list = data.split(" ")
    phones_set = set(phones_list)
    return phones_set


def _create_phones_inventory(data):
    phone_inv = set()
    columns = ["source", "target"]
    data.columns = columns
    assert len(data.columns) == 2
    assert 'source' in data.columns
    assert 'target' in data.columns
    targets = data['target'].to_list()
    for targ in targets:
        phone_set = _create_set(targ)
        phone_inv.update(phone_set)
    return phone_inv


def check_phone_set_difference(gold_phone_set: set, data_phone_set: set):
    return data_phone_set.difference(gold_phone_set)


def main(args):
    phones_path = os.path.join(PHONES_DIR, f"{args.lang}.txt")
    lang_path = find_directory_in_subdirectories(TSV_DIR, args.lang)
    # Get phone set.
    with open(phones_path, "r", encoding="utf8") as src:
        phones = src.readlines()
        phones_set = set([phone.rstrip() for phone in phones])
    print(f"gold phones: {phones_set}")
    #Process Train datasets.
    train_path = f"{lang_path}/train/{args.lang}_train.tsv"
    train_data = pd.read_csv(train_path, sep="\t")
    train_phone_set = _create_phones_inventory(train_data)
    print(f"data phones: {train_phone_set}")
    phone_dif = check_phone_set_difference(phones_set, train_phone_set)
    print(f"set difference: {phone_dif}")
    # Process Val datasets.
    val_path = f"{lang_path}/val/{args.lang}_val.tsv"
    test_path = f"{lang_path}/test/{args.lang}_test.tsv"
    # Read tsv files.



if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-lang", help="string identifying the language being processed.")
    main(parser.parse_args())