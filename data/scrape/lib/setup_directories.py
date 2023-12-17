#!/usr/bin/env python3

import argparse
from distutils.command.config import config
import os
import shutil
import yaml

from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True, help="path to config file.")
    args = parser.parse_args()
    # Load config file.
    with open(args.config, "r") as src:
        try:
            conf = yaml.safe_load(src)
        except yaml.YAMLError as exc:
            print(exc)
    TSV_DIR = os.path.join(ROOT_DIR, conf["tsv_dir"])

    # Create directory for langauges if they don't exist.
    for langauge_groups in tqdm(conf["languages"], desc="Creating Directories"):
        for orth, langs in langauge_groups.items():
            orth_path = os.path.join(TSV_DIR, orth)
            if not os.path.exists(orth_path):
                os.makedirs(orth_path)
            try:
                for lang in langs:
                    lang_path = os.path.join(TSV_DIR, orth, lang)
                    if not os.path.exists(lang_path):
                        os.makedirs(lang_path)
                    # Create data set folders.
                    raw_data_path = os.path.join(TSV_DIR, orth, lang, "raw")
                    train_data_path = os.path.join(TSV_DIR, orth, lang, "train")
                    val_data_path = os.path.join(TSV_DIR, orth, lang, "val")
                    test_data_path = os.path.join(TSV_DIR, orth, lang, "test")
                    if not os.path.exists(raw_data_path):
                        os.makedirs(raw_data_path)
                    if not os.path.exists(train_data_path):
                        os.makedirs(train_data_path)
                    if not os.path.exists(val_data_path):
                        os.makedirs(val_data_path)
                    if not os.path.exists(test_data_path):
                        os.makedirs(test_data_path)
            except TypeError as error:
                print(error)

    
