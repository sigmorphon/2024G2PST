# Task 1: Fourth SIGMORPHON Shared Task on Grapheme-to-Phoneme Conversions
In this task, participants will create computational models that map native orthography lemmas ("graphemes") to valid IPA transcription of that word's pronunciation. This task is crucial for speech processing, namely speech recognition and text-to-speech synthesis.

Please sign up for the mailing list [here](sigmorphon-g2p-shared-task-2024@google) by clicking the button labeled "Ask to join group".

## Results
Final results will be reported here by August 15, 2024.  System papers and a summary of the task will appear in the SIGMORPHON 2024 proceedings.

## Data
### Source
The data is scraped from [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Main_Page) using [WikiPron](https://github.com/kylebgorman/wikipron) (Lee et al. 2020).

### Format
Training and development data are UTF-8-encoded tab-separated values files. Each example occupies a single line and consists of a grapheme sequence—a sequence of [NFC](https://en.wikipedia.org/wiki/Unicode_equivalence#Normal_forms) Unicode codepoints—a tab character, and the corresponding phone sequence, a roughly-phonemic IPA, tokenized using the [segments]() library. The following shows three lines of Romanian data:

### Subtasks
There are three subtasks, which will be scored separately. Participant teams may submit as many systems as they want to as many subtasks as they want.

In all three subtasks, the data is randomly split into training (80%), development (10%), and testing (10%) data.

#### Subtask 1: Multilingual G2P (Shared Orthography):

Participants will be provided four multilingual training sets. Each dataset will be composed of languages utilizing the same orthography (Roman, Cyrillic, Devanagari, Arabic). For evaluation, each training set will be paired with a test set, with each test set composed of samples from the training languages along with two to three languages unseen in training but utilizing the same orthography as seen in training. Models in this track will be tasked to evaluate per training-test dataset pairing (i.e. models are trained on one orthography at a time). 

#### Subtask 2: Multilingual G2P (Restricted Orthography):

Participants will be provided four multilingual training sets using the same script constraints as seen in Task 1. Provided these datasets, participants will be allowed to train over the concatenation of all data (or subset of) for single evaluation on the concatenation of all test datasets in Task 1. That is, the subtask will evaluate model ability to utilize mixed-orthography scripts for downstream evaluation.

#### Subtask 3: Multilingual G2P (Unknown Orthograph):

This task will function similarly to Task 2 but all unseen languages in the test set will be replaced with languages with orthographies distinct from all other scripts present in the preceding two tasks. For fairness, script systems will be chosen such that they are functionally similar to the training scripts (e.g. the unseen Devanagari languages will be replaced with other North Indic abugida).

## Evaluation
The metric used to rank systems is word error rate (WER), the percentage of words for which the hypothesized transcription sequence does not match the gold transcription. This value, in accordance with common practice, is a decimal value multiplied by 100 (e.g.: 13.53). In the medium- and low-frequency tasks, WER is macro-averaged across all ten languages. We provide two Python scripts for evaluation:

[evaluate.py]() computes the WER for one language.
[evaluate_all.py]() computes per-language and average WER across multiple languages.

## Submission
Please submit your results in the two-column (grapheme sequence, tab-character, tokenized phone sequence) TSV format, the same one used for the training and development data. If you use an internal representation other than NFC, you must convert back before submitting.

Please use [this email form]() to submit your results.
## Timeline
* January 15, 2024: Data collection is complete, and data is released to participants
* February 15, 2024: Baseline systems released to participants
* May 15, 2024: Test data is available for participants
* May 31, 2024: Final Submissions are due
* June 3, 2024: Results announced to participants
* June 22, 2024: System papers due for review
* July 31, 2024: Reviews back to participants
* August 15, 2024: CR deadline; task paper due from organizers.

## Baseline
Baseline results and discription will be released on February 15.

## Comparison with 2022 shared task
* Transfer languages are provided.
* There are new languages.
* The three subtasks have changed, organized around research questions.
* There are surprise languages.
* The data been subjected to novel quality-assurance procedures.
  
## Organizers
The task is organized by members of the Computational Linguistics Lab at the [Graduate Center, City University of New York](https://www.gc.cuny.edu/) and the [University of British Columbia]().

## Liscensing
The code is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The data is released under the [Creative Commons Attribution-ShareAlike 3.0 Unported License](https://creativecommons.org/licenses/by-sa/3.0/legalcode) inherited from Wiktionary itself.

## Referencing
Girrbach, L. 2023. [SIGMORPHON 2022 Shared Task on Grapheme-to-Phoneme Conversion Submission Description: Sequence Labelling for G2P](https://aclanthology.org/2023.sigmorphon-1.28/). Proceedings of the 20th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology, pages 239–244.

Lee, J. L, Ashby, L. F.E., Garza, M. E., Lee-Sikka, Y., Miller, S., Wong, A., McCarthy, A. D., and Gorman, K. 2020. [Massively multilingual pronunciation mining with WikiPron](). In Proceedings of the 12th Language Resources and Evaluation Conference, pages 4223-4228.

Makarov, P., and Clematide, S. 2018. [Imitation learning for neural morphological string transduction](). In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2877-2882.

Makarov, P., and Clematide, S. 2020. [CLUZH at SIGMORPHON 2020 shared task on multilingual grapheme-to-phoneme conversion](). In Proceedings of the 17th SIGMORPHON Workshopon Computational Research in Phonetics, Phonology, and Morphology, pages 171-176.
