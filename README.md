## Requirements
* Python 3.8
* PyTorch 1.10.1
* torchtext 0.6
* CUDA & CuDNN to run on GPU (recommended)
* javalang 0.13
* learn2learn 0.1.5
* tqdm 4.47
* numpy 1.18.1
* NLTK 3.5
* py-rouge 1.1

## Data
Original data we used is from https://github.com/xing-hu/TL-CodeSum and https://github.com/EdinburghNLP/code-docstring-corpus. This repository contains data processing script for the former. Please put the raw data into the **data/original** folder. The code to shuffle and remove the duplicate examples is not included so you have to manually do this if required.

## Preprocessing
* Run `python Data.py` to filter & tokenize the codes and comments. Comments with less than 2 tokens after tokenization will be marked for discard.

## Build & Train a basic code summarizer
* Run `python Train.py` to train a standard seq2seq model based on biLSTM as the basic code summarizer for MLCS.

## Retrieve similar code snippets
* Run `python Retrieval.py` to retrieve similar examples for each code snippet in the dataset using the hidden vectors computed by the seq2seq model. The index of retrieved examples and the distances will be saved under **data/retrieved**. 

## Optimize the basic model into a meta learner and then evaluate
* Run `python Main.py` to start the training procedure of MLCS. After training, the performance on the testing data will be evaluated. The predictions will be saved under the **predicts** folder and the scores will be printed.

## Change the Parameters
All parameters are defined in the beginning of the .py files. We will make the scripts easier to use in the future (e.g., by using argparse).
