#!/usr/bin/env python3
"""Augment datasets for Urdu, Nepali, and Marathi."""

import argparse
import glob
import os
import time
from tqdm import tqdm
import yaml


import openai
from openai import OpenAI
from datasets import load_dataset
import unicodedataplus
import random
import spacy
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair

from typing import List, Dict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NEPALI_DIR = os.path.join(ROOT_DIR, "tsv", "devanagari", "nep", "raw", "root_pos_rules.csv")
# URDU_PATHS = os.path.join(ROOT_DIR, "tsv", "abjad", "urd", "raw")
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")



def _check_script(word: str, orth_id: str) -> bool:
    for char in word:
        if unicodedataplus.script(char) != orth_id:
            return False
    return True


def _create_sample_examples(src_path: str, n = 50, seed = 42) -> List:
    
    with open(src_path, "r", encoding="utf8") as src:
        lines = src.readlines()
        lines = [line.rstrip() for line in lines]
        random.seed(seed)
        random_lines = random.sample(lines, n)
        random_lines = ('\n'.join(random_lines))
        return random_lines


def _create_google_sample_examples(src_path: str, n = 50, seed = 42) -> List:
    examples = []
    with open(src_path, "r", encoding="utf8") as src:
        lines = src.readlines()
        lines = [line.rstrip() for line in lines]
        random.seed(seed)
        random_lines = random.sample(lines, n)
    for line in random_lines:
        line = line.split("\t")
        src = line[0]
        trg = line[1]
        example = InputOutputTextPair(src, trg)
        examples.append(example)
    return examples


def extract_urdu_samples(orth_id: str =  "Arabic", n: int = 3000, seed: int = 42, sample_set_size = 50) -> set:
    sample_sets = []
    samples_set = set()
    pattern = f'{URDU_PATHS}/ur_udtb-ud-*.conllu'
    tsv_path = f'{URDU_PATHS}/urd.tsv'
    urdu_files = glob.glob(pattern)
    with open(tsv_path, "r", encoding="utf8") as src:
        for line in src:
            line = line.split('\t')
            lemma = line[0]
            samples_set.add(lemma)
    for file in urdu_files:
        with open(file, "r", encoding="utf8") as src:
            for line in tqdm(src):
                lst = line.split("\t")
                if len(lst) > 2:
                    if not lst[2].isnumeric():
                        if lst[3] != "PUNCT":
                            if _check_script(lst[2],orth_id):
                                lemma = lst[2]
                                samples_set.add(lemma)
    random.seed(seed)
    # Create 3000 samples
    selected_samples = random.sample(samples_set, n)
    # Subset samples to 500 per loop for
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
    MaxToken=None): 
    # using OpenAI's Completion module that helps execute  
    # any tasks involving text 
    client = OpenAI(api_key=auth_key)

    message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": gpt_user_prompt.format(language = language, examples = examples, words = words)}]
    temperature=0.1
    max_tokens=MaxToken
    frequency_penalty=0.0

    response = client.chat.completions.create(
        model="gpt-4",
        messages = message,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty
    )
    return response.choices[0].message.content


def get_palm_ipa_transcription(
    language: str, 
    examples: List[str],
    word: str,
    assistant_prompt: str,
    user_prompt: str,
    project_id: str, 
    location: str,
    MaxToken: int = 50):
    
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    
    # Load the model
    chat_model = ChatModel.from_pretrained("text-bison")

    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": 0.1,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": MaxToken,  # Token limit determines the maximum amount of text output.
        "top_p": 0.99,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 1,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    chat = chat_model.start_chat(
        context=assistant_prompt,
        examples=examples,
    )
    user_response = user_prompt.format(language = language, examples = examples, word = word)
    response = chat.send_message(
        user_response, **parameters
    )
 

    return response.text


def main(args):
    # Load config
    with open(CONFIG_PATH) as src:
        config = yaml.safe_load(src)
    # Authenticate environment
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config['auth_key']
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

    # Retrieve Urdu Language Data
    if args.lang == "urd":
        samples = extract_urdu_samples()
        # print(samples)
        # track_samples = dict()
        pattern = f'{config["URDU_RAW"]}/ur_udtb-ud-*.conllu'
        tsv_path = os.path.join(ROOT_DIR, config["URDU_SAMPLE"])
        outpath = os.path.join(ROOT_DIR, config["URDU_OUTPATH"])
        examples = _create_sample_examples(tsv_path)
        with open(outpath, "a", encoding="utf8") as sink:
            for sample in tqdm(samples):
                # Get GPT Transcriptions
                transcription = get_gpt_ipa_transcription(
                    gpt_assistant_prompt=gpt_assistant_prompt, 
                    gpt_user_prompt= gpt_user_prompt, 
                    language = "Urdu", 
                    examples = examples, 
                    words = sample,
                    auth_key=config["OPEN_AI_KEY"], 
                    MaxToken=None)
                time.sleep(3)
                print(transcription.rstrip(),file=sink)
    # Retrieve Nepali Language Data
    if args.lang == "nep":
        samples = extract_urdu_samples()
        # print(samples)
        # track_samples = dict()
        tsv_path = os.path.join(ROOT_DIR, config["NEP_SAMPLE"])
        outpath = os.path.join(ROOT_DIR, config["NEP_OUTPATH"])
        examples = _create_sample_examples(tsv_path)
        with open(outpath, "a", encoding="utf8") as sink:
            for sample in tqdm(samples):
                # Get GPT Transcriptions
                transcription = get_gpt_ipa_transcription(
                    gpt_assistant_prompt=gpt_assistant_prompt, 
                    gpt_user_prompt= gpt_user_prompt, 
                    language = "Nepali", 
                    examples = examples, 
                    words = sample,
                    auth_key=config["OPEN_AI_KEY"], 
                    MaxToken=None)
                time.sleep(3)
                print(transcription.rstrip(),file=sink)
    # Retrieve Marathi Language Data
    if args.lang == "mar":
        samples = extract_urdu_samples()
        # print(samples)
        # track_samples = dict()
        tsv_path = os.path.join(ROOT_DIR, config["MAR_SAMPLE"])
        outpath = os.path.join(ROOT_DIR, config["MAR_OUTPATH"])
        examples = _create_sample_examples(tsv_path)
        with open(outpath, "a", encoding="utf8") as sink:
            for sample in tqdm(samples):
                # Get GPT Transcriptions
                transcription = get_gpt_ipa_transcription(
                    gpt_assistant_prompt=gpt_assistant_prompt, 
                    gpt_user_prompt= gpt_user_prompt, 
                    language = "Marathi", 
                    examples = examples, 
                    words = sample,
                    auth_key=config["OPEN_AI_KEY"], 
                    MaxToken=None)
                time.sleep(3)
                print(transcription.rstrip(),file=sink)
    if args.lang == "pus":
        samples = extract_urdu_samples()
        tsv_path = os.path.join(ROOT_DIR, config["PUS_SAMPLE"])
        outpath = os.path.join(ROOT_DIR, config["PUS_OUTPATH"])
        examples = _create_sample_examples(tsv_path)
        with open(outpath, "a", encoding="utf8") as sink:
            for sample in tqdm(samples):
                # Get GPT Transcriptions
                transcription = get_gpt_ipa_transcription(
                    gpt_assistant_prompt=gpt_assistant_prompt, 
                    gpt_user_prompt= gpt_user_prompt, 
                    language = "Pashto", 
                    examples = examples, 
                    words = sample,
                    auth_key=config["OPEN_AI_KEY"], 
                    MaxToken=None)
                time.sleep(3)
                print(transcription.rstrip(),file=sink)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-lang", required=True, help="lang code for processing")
    main(parser.parse_args())