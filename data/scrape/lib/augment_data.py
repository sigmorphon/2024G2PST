#!/usr/bin/env python3
"""Augment datasets for Urdu, Nepali, and Marathi."""

import argparse
import glob
import math
import os
import time
from tqdm import tqdm
import yaml

from datasets import load_dataset
import pandas as pd
import mahaNLP
from mahaNLP.preprocess import Preprocess
from mahaNLP.tokenizer import Tokenize
import numpy as np
import openai
from openai import OpenAI
import random
import unicodedataplus
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair

from typing import List, Dict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")


def _tokenize_marathi(sent: str):
    model = Tokenize()
    tokenized_sent = model.word_tokenize(sent, punctuation = False)
    return tokenized_sent


def _filter_marathi(sent: str):
    model = Preprocess()
    remove_url = model.remove_url(sent)
    filtered_sentence = model.remove_stopwords(remove_url)
    return filtered_sentence


def _check_script(word: str, orth_id: str) -> bool:
    """
    Checks if all characters in a given word belong to a specific script as identified 
    by the orthographic identifier.

    Args:
        word (str): The word whose characters are to be checked.
        orth_id (str): The orthographic identifier representing a script (e.g., 'Latin', 'Arabic'). 
            This is used to check each character of the word against.

    Returns:
        bool: True if all characters in the word belong to the script specified by orth_id, False otherwise.
    """
    for char in word:
        if unicodedataplus.script(char) != orth_id:
            return False
    return True


def _create_sample_examples(src_path: str, n=50, seed=42) -> List:
    """
    Selects a random sample of lines from a text file and returns them as a single string.

    Args:
        src_path (str): The path to the source text file from which lines will be sampled.
        n (int, optional): The number of lines to sample from the source file. Defaults to 50.
        seed (int, optional): The seed value for the random number generator to ensure 
            reproducibility of the sample selection. Defaults to 42.

    Returns:
        List: A string where each element is a sampled line from the source file, 
            combined into a single string separated by newline characters.
    """
    with open(src_path, "r", encoding="utf8") as src:
        lines = src.readlines()
        lines = [line.rstrip() for line in lines]
        random.seed(seed)
        random_lines = random.sample(lines, n)
        random_lines = "\n".join(random_lines)
        return random_lines


def extract_urdu_samples(
    orth_id: str = None,
    n: int = 3000,
    seed: int = 42,
    sample_set_size=50,
    pattern=None
) -> List[str]:
    """
    Extracts a set of Urdu language samples and organizes them into smaller subsets
    for feeding to GPT-API for transcription.

    Args:
        orth_id (str, optional): Orthographic identifier used to check the script of the samples.
            If None, the script check is skipped. Defaults to None.
        n (int, optional): Total number of samples to select randomly from the collected samples.
            Defaults to 3000.
        seed (int, optional): Seed value for random sample selection to ensure reproducibility.
            Defaults to 42.
        sample_set_size (int, optional): Number of samples in each subset created from the total samples.
            Defaults to 50.
        pattern (str, optional): Glob pattern to identify and load Urdu text files. If None, no files are loaded.
            Defaults to None.
        tsv_path (str, optional): Path to a TSV file containing Urdu samples. If None, the file is not loaded.
            Defaults to None.

    Returns:
        List: List of strings, where each string is a newline-separated list of Urdu samples.
             Each string in the set represents a subset of the total samples.
    """
    sample_sets = []
    samples_set = set()
    urdu_files = glob.glob(pattern)
    for file in urdu_files:
        with open(file, "r", encoding="utf8") as src:
            for line in tqdm(src):
                lst = line.split("\t")
                if len(lst) > 2:
                    if not lst[2].isnumeric():
                        if lst[3] != "PUNCT":
                            if _check_script(lst[2], orth_id):
                                lemma = lst[2]
                                samples_set.add(lemma)
    random.seed(seed)
    # Create 3000 samples
    selected_samples = random.sample(samples_set, n)
    # Subset samples to 50 per loop for
    # GPT transcription and write to list.
    sample_loops = int(n / sample_set_size)
    for i in range(sample_loops):
        samples = random.sample(selected_samples, sample_set_size)
        one_column_sample = "\n".join(samples)
        sample_sets.append(one_column_sample)
    return sample_sets


def extract_pus_samples(
    source: str,
    orth_id: str = None,
    n: int = 4000,
    seed: int = 42,
    sample_set_size=50,
    ) -> List[str]:

    df = pd.read_csv(source)
    sample_sets = []
    selected_samples = set(df.iloc[:, 0].to_list())
    # Subset samples to 50 per loop for
    # GPT transcription and write to list.
    sample_loops = math.ceil(len(selected_samples) / sample_set_size)
    for i in range(sample_loops):
        samples = random.sample(selected_samples, sample_set_size)
        one_column_sample = "\n".join(samples)
        sample_sets.append(one_column_sample)
    return sample_sets


def extract_mar_samples(
    orth_id: str = None,
    n: int = 4000,
    seed: int = 42,
    sample_set_size=50,
    pattern=None,
    excel_file=None) -> List[str]:
    sample_sets = []
    samples_set = set()
    files = glob.glob(pattern)
    # Sample UD file set
    for file in files:
        with open(file, "r", encoding="utf8") as src:
            for line in tqdm(src):
                lst = line.split("\t")
                if len(lst) > 2:
                    if not lst[2].isnumeric():
                        if lst[3] != "PUNCT":
                            if _check_script(lst[2], orth_id):
                                lemma = lst[2]
                                samples_set.add(lemma)
    # Sample hate speech set
    # loads the dataframe into 'dataset'
    # dataset_1 = load_datasets('mahaSent')
    excel_mar = pd.read_excel(excel_file)
    excel_sents = excel_mar["text"].to_list()
    for sent in excel_sents:
        tokenized_sent = _filter_marathi(sent)
        for token in tokenized_sent:
            if _check_script(token, orth_id):
                if len(token) >= 3 and len(token) <= 9:
                    samples_set.add(token)

    random.seed(seed)
    # Create 3000 samples
    selected_samples = random.sample(samples_set, n)
    # Subset samples to 50 per loop for
    # GPT transcription and write to list.
    sample_loops = int(n / sample_set_size)
    for i in range(sample_loops):
        samples = random.sample(selected_samples, sample_set_size)
        one_column_sample = "\n".join(samples)
        sample_sets.append(one_column_sample)
    return sample_sets
    


def extract_nep_samples(
    source: str,
    orth_id: str = None,
    n: int = 4000,
    seed: int = 42,
    sample_set_size=50,
    ) -> List[str]:
    df = pd.read_csv(source)
    sample_sets = []
    sample_words = df.sample(n=n,random_state=seed)
    selected_samples = set(sample_words["word"])
    # Subset samples to 50 per loop for
    # GPT transcription and write to list.
    sample_loops = int(n / sample_set_size)
    for i in range(sample_loops):
        samples = random.sample(selected_samples, sample_set_size)
        one_column_sample = "\n".join(samples)
        sample_sets.append(one_column_sample)
    return sample_sets



def get_gpt_ipa_transcription(
    language,
    examples,
    words: str,
    gpt_assistant_prompt,
    gpt_user_prompt,
    auth_key,
    MaxToken=None,
):
    """
    Generates IPA (International Phonetic Alphabet) transcriptions for given words using GPT-4.

    Args:
        language: The language of the words for which IPA transcriptions are requested.
        examples: Examples of transcriptions or related context to guide the GPT model.
        words (str): A string of newline separated words to be transcribed into IPA.
        gpt_assistant_prompt: The initial prompt or statement by the assistant to set the context for the GPT model.
        gpt_user_prompt: A formatted string representing the user's request or question to the GPT model. 
            This string should contain placeholders for 'language', 'examples', and 'words'.
        auth_key: Authentication key for accessing OpenAI's API.
        MaxToken (int, optional): Maximum number of tokens to generate in the response. 
            If None, the default maximum is used.

    Returns:
        str: The content of the response from GPT-4, expected to be the IPA transcription of the provided words.
    """
    # using OpenAI's Completion module that helps execute
    # any tasks involving text
    client = OpenAI(api_key=auth_key)

    message = [
        {"role": "assistant", "content": gpt_assistant_prompt},
        {
            "role": "user",
            "content": gpt_user_prompt.format(
                language=language, examples=examples, words=words
            ),
        },
    ]
    temperature = 0.1
    max_tokens = MaxToken
    frequency_penalty = 0.0

    response = client.chat.completions.create(
        model="gpt-4",
        messages=message,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )
    return response.choices[0].message.content


def main(args):
    # Load config
    with open(CONFIG_PATH) as src:
        config = yaml.safe_load(src)
    # Authenticate environment
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config["auth_key"]
    # Create prompt to pass to GPT for transcription.
    gpt_assistant_prompt = """
    You are a helpful IPA transcription assistant. 
    """
    gpt_user_prompt = """
    Provide a direct IPA transcription for the {language} words given in double quotations below.
    ""{words}"". 
    Format your response as a two column TSV, following the format of the example TSV given
    between the --- delimiters below. For each transciption, provide ONLY the original word
    as the value of the first column and provide ONLY your space separated 
    transcription of the original word as the value of the second column. 
    ---
    {examples}
    ---
    """
    #TODO: make one function with expected data format.
    # Retrieve Urdu Language Data
    if args.lang == "urd":
        urdu_path = config["URDU_RAW"]
        samples = extract_urdu_samples(
            orth_id="Arabic",
            pattern=f"{urdu_path}/ur_udtb-ud-*.conllu"
        )
        tsv_path = os.path.join(ROOT_DIR, config["URDU_SAMPLE"])
        outpath = os.path.join(ROOT_DIR, config["URDU_OUTPATH"])
        examples = _create_sample_examples(tsv_path)
        with open(outpath, "a", encoding="utf8") as sink:
            for sample in tqdm(samples):
                # Get GPT Transcriptions
                transcription = get_gpt_ipa_transcription(
                    gpt_assistant_prompt=gpt_assistant_prompt,
                    gpt_user_prompt=gpt_user_prompt,
                    language="Urdu",
                    examples=examples,
                    words=sample,
                    auth_key=config["OPEN_AI_KEY"],
                    MaxToken=None,
                )
                time.sleep(3)
                print(transcription.rstrip(), file=sink)
    # Retrieve Nepali Language Data
    if args.lang == "nep":
        src = os.path.join(ROOT_DIR, config["NEP_SAMPLE"])
        samples = extract_nep_samples(
            source = src
        )  # Write function to get samples
        # track_samples = dict()
        tsv_path = os.path.join(ROOT_DIR, config["NEP_EXAMPLES"])
        outpath = os.path.join(ROOT_DIR, config["NEP_OUTPATH"])
        examples = _create_sample_examples(tsv_path)
        with open(outpath, "a", encoding="utf8") as sink:
            for sample in tqdm(samples):
                # Get GPT Transcriptions
                transcription = get_gpt_ipa_transcription(
                    gpt_assistant_prompt=gpt_assistant_prompt,
                    gpt_user_prompt=gpt_user_prompt,
                    language="Nepali",
                    examples=examples,
                    words=sample,
                    auth_key=config["OPEN_AI_KEY"],
                    MaxToken=None,
                )
                time.sleep(3)
                print(transcription.rstrip(), file=sink)
    # Retrieve Marathi Language Data
    if args.lang == "mar":
        mar_path = os.path.join(ROOT_DIR, config["MAR_RAW"])
        mar_excel_path = os.path.join(ROOT_DIR, config["MAR_XLSX"])
        samples = extract_mar_samples(
            orth_id="Devanagari",
            pattern=f"{mar_path}/mr_ufal-ud-*.conllu",
            excel_file=mar_excel_path)
        # track_samples = dict()
        tsv_path = os.path.join(ROOT_DIR, config["MAR_EXAMPLES"])
        outpath = os.path.join(ROOT_DIR, config["MAR_OUTPATH"])
        examples = _create_sample_examples(tsv_path)
        with open(outpath, "a", encoding="utf8") as sink:
            for sample in tqdm(samples):
                # Get GPT Transcriptions
                transcription = get_gpt_ipa_transcription(
                    gpt_assistant_prompt=gpt_assistant_prompt,
                    gpt_user_prompt=gpt_user_prompt,
                    language="Marathi",
                    examples=examples,
                    words=sample,
                    auth_key=config["OPEN_AI_KEY"],
                    MaxToken=None,
                )
                time.sleep(3)
                print(transcription.rstrip(), file=sink)
    if args.lang == "pus":
        sample_source = os.path.join(ROOT_DIR, config["PUS_SAMPLE"])
        samples = extract_pus_samples(sample_source)  # Write function to get samples
        tsv_path = os.path.join(ROOT_DIR, config["PUS_EXAMPLES"])
        outpath = os.path.join(ROOT_DIR, config["PUS_OUTPATH"])
        examples = _create_sample_examples(tsv_path)
        with open(outpath, "a", encoding="utf8") as sink:
            for sample in tqdm(samples):
                # Get GPT Transcriptions
                transcription = get_gpt_ipa_transcription(
                    gpt_assistant_prompt=gpt_assistant_prompt,
                    gpt_user_prompt=gpt_user_prompt,
                    language="Pashto",
                    examples=examples,
                    words=sample,
                    auth_key=config["OPEN_AI_KEY"],
                    MaxToken=None,
                )
                time.sleep(3)
                print(transcription.rstrip(), file=sink)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "-lang", required=True, help="lang code for processing"
    )
    main(parser.parse_args())
