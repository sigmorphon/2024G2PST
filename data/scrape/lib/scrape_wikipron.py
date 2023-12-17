#!/usr/bin/env python3
"""Scrape all relevant data source."""

import argparse
import os
import shutil
from tqdm import tqdm
import yaml

import numpy as np
import pandas as pd
import wikipron

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV_DIR = os.path.join(ROOT_DIR, "tsv")

def scrape_wikipron(iso_code: str, dialect: str = None) -> pd.DataFrame:
    source = []
    target = []
    if dialect is not None:
      config = wikipron.Config(key=iso_code, dialect=dialect) # Default options.
      for word, pron in tqdm(wikipron.scrape(config), desc=f"Scraping words for {iso_code}"):
          source.append(word)
          target.append(pron)
      return pd.DataFrame({"source": source, "target": target})
    else:
      config = wikipron.Config(key=iso_code, dialect=dialect) # Default options.
      for word, pron in tqdm(wikipron.scrape(config), desc=f"Scraping words for {iso_code}"):
          source.append(word)
          target.append(pron)
      return pd.DataFrame({"source": source, "target": target})


def find_root_in_subdirectories(directory, filename):
  """
  Searches all subdirectories of a directory for a file and returns its full path.

  Args:
    directory: The directory to start searching from.
    filename: The name of the file to search for.

  Returns:
    The full path to the file if found, or None if not found.
  """

  for root, _, _ in os.walk(directory):
    raw_path = f"{filename}/raw"
    if raw_path in root:
      # Found the file! Return its full path.
      return root

  # File not found in any subdirectory.
  return None


def main(args):
    # Check if language data has already been scraped.
    lang_path = find_root_in_subdirectories(TSV_DIR, args.iso_code)
    tsv_name = f"{args.iso_code}.tsv"
    path_to_lang_data = os.path.join(lang_path, tsv_name)
    if os.path.exists(path_to_lang_data):
      print(f"{tsv_name} already exists. Delete the file to redownload.")
    else:
      # Scrape Wikipron.
      df = scrape_wikipron(args.iso_code, dialect = args.dialect)
      # Write to language path.
      df.to_csv(path_to_lang_data, sep="\t", index=False, header=False)
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("iso_code", help="a three-letter ISO 639-3 language code, e.g., fra for French")
    parser.add_argument("-dialect", help="dialect id for wikipron language.")
    main(parser.parse_args())