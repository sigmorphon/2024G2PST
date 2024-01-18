"""Split data into train, dev, and test."""

import json
import logging
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHONES_DIR = os.path.join(ROOT_DIR, "phones")
TSV_DIR = os.path.join(ROOT_DIR, "tsv")
LOG_PATH = os.path.join(ROOT_DIR, "logs", "change_phones.log")

logging.basicConfig(filename=LOG_PATH, filemode='w', format='%(message)s')


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

  for root, dirs, _ in os.walk(directory):
    if subdirectory in dirs:
      # Found the file! Return its full path.
      return os.path.join(root, subdirectory)

  # File not found in any subdirectory.
  return None

def _find_phone(phone, phone_schema: json):
    for gold_phone, phonetic_values in phone_schema.items():
        if phone in phonetic_values:
            return gold_phone
    return phone


def _filter_phones(transcription: str, phone_schema: json, language: str = None):
    clean_phones = []
    phones = transcription.split(" ")
    for phone in phones:
        clean_phone = _find_phone(phone, phone_schema)
        if language == "rus":
            filter_superscripts = ["⁽ʲ", "⁾ː", "⁾"]
            for super in filter_superscripts:
                if super in clean_phone:
                    clean_phone.split(super)
                    clean_phone = clean_phone[0]
        clean_phones.append(clean_phone)

    gold_transcription = " ".join(clean_phones)
    if gold_transcription != transcription:
        substitution = f"{transcription} -> {gold_transcription}\n"
        logging.info(substitution)
    return gold_transcription


def substitute_phones(df: pd.DataFrame, phone_schema: json, language: str = None) -> pd.DataFrame:
    df["target"] = df["target"].apply(lambda x: _filter_phones(x, phone_schema))
    return df
    


def main(args):
    LANG_PATH = find_file_in_subdirectories(TSV_DIR, args.language)
    PHONES_JSON_PATH = os.path.join(PHONES_DIR, f"{args.language}.json")
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
                data = pd.read_csv(src, sep="\t").sample(n=3000, replace=False, random_state=42)
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
    # Clean data if needed.
    if args.clean == "y":
        logging.info(f"{args.language} CHANGE LOG")
        logging.info(f"----------------")
        with open(PHONES_JSON_PATH, 'r', encoding="utf8") as src:
            gold_phone_set = json.load(src)
        clean_train_set = substitute_phones(train_set, gold_phone_set)
        clean_val_set = substitute_phones(val_set, gold_phone_set)
        clean_test_set = substitute_phones(test_set, gold_phone_set)
        # Write sets path. 
        train_path = f"{LANG_PATH}/train/{args.language}_train.tsv"
        val_path = f"{LANG_PATH}/val/{args.language}_val.tsv"
        test_path = f"{LANG_PATH}/test/{args.language}_test.tsv"
        # if not os.path.exists(train_path):
        clean_train_set.to_csv(train_path, header=False, index=False, sep="\t")
        # if not os.path.exists(val_path):
        clean_val_set.to_csv(val_path, header=False, index=False, sep="\t")
        # if not os.path.exists(test_path):
        clean_test_set.to_csv(test_path, header=False, index=False, sep="\t")
        logging.info(f"\n")
    else:
        logging.info(f"{args.language} CHANGE LOG")
        logging.info(f"----------------")
        # Write to file.
        train_path = f"{LANG_PATH}/train/{args.language}_train.tsv"
        val_path = f"{LANG_PATH}/val/{args.language}_val.tsv"
        test_path = f"{LANG_PATH}/test/{args.language}_test.tsv"
        # if not os.path.exists(train_path):
        train_set.to_csv(train_path, header=False, index=False, sep="\t")
        # if not os.path.exists(val_path):
        val_set.to_csv(val_path, header=False, index=False, sep="\t")
        # if not os.path.exists(test_path):
        test_set.to_csv(test_path, header=False, index=False, sep="\t")
        logging.info(f"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--infile", help="path to ingestion file for data splitting.")
    parser.add_argument("--language", help="string identifying the language being processed.")
    parser.add_argument("-dedupe", help="y/n indicating whether file needs deduping.")
    parser.add_argument("-clean", help="y/n indicating if file needs to be cleaned or not")
    main(parser.parse_args())