import os
import argparse

import numpy as np
import lightning as pl
import lightning
import torch
import transformers

from transformers.models.t5.tokenization_t5_fast import T5Tokenizer
from transformers import \
            T5ForConditionalGeneration, \
                T5TokenizerFast as T5Tokenizer
                

from torch.utils.data import Dataset, DataLoader

from l4nlp.t5_summarizer import Summarizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_name", type=str, default="t5-base",\
            help="model name, options: t5-small, t5-base")
    parser.add_argument("-i", "--input_filepath", type=str, default="my_file.txt",\
            help="filepath to .txt to summarize")
    parser.add_argument("-d", "--device", type=str, default="gpu",\
            help="hardware device to use for training")
    parser.add_argument("-p", "--parameters_filepath", type=str, default="model_temp.pt",\
            help="filepath to model parameters")
    parser.add_argument("-w", "--beam_width", type=int, default=1,\
            help="beam width to use")



    args = parser.parse_args()
    model_name = args.model_name
    input_filepath = args.input_filepath
    beam_width = args.beam_width
    parameters_filepath = args.parameters_filepath

    model = Summarizer(model_name=model_name)
    model.number_beams = beam_width
    model.text_max_length = 512
    model.summary_max_length = 64
    model.load_state_dict(torch.load(parameters_filepath))

    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)

    with open(input_filepath, "r") as f:
        text = f.readlines()

    my_summary = model.summarize(text=text, tokenizer=tokenizer)
    print(f"summary: \n\n {my_summary}")

    output_filepath = os.path.splitext(input_filepath)[0] + "_summary.txt"

    with open(output_filepath, "w") as f:
        f.writelines(my_summary)
