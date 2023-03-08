import os
import argparse
import json

import numpy as np
import lightning as pl
import lightning
import torch
import transformers

from transformers.models.t5.tokenization_t5_fast import T5Tokenizer
from transformers import \
            T5ForConditionalGeneration, \
                T5TokenizerFast as T5Tokenizer
                

import pandas as pd
from torch.utils.data import Dataset, DataLoader

class SummaryDataset(Dataset):

    def __init__(self,\
                data, tokenizer,\
                text_max_length=512,\
                summary_max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_length = text_max_length
        self.summary_max_length = summary_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data_row = self.data.iloc[index]

        text = data_row["text"]
        summary = data_row["summary"]

        text_encoding = self.tokenizer(\
                text,\
                max_length=self.text_max_length,\
                padding="max_length",\
                return_attention_mask=True,\
                add_special_tokens=True,\
                return_tensors="pt",\
                truncation=True)

        summary_encoding = self.tokenizer(\
                summary,\
                max_length=self.summary_max_length,\
                padding="max_length",\
                return_attention_mask=True,\
                add_special_tokens=True,\
                return_tensors="pt",\
                truncation=True)

        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100

        return {"text": text,\
                "input_ids": text_encoding["input_ids"],\
                "summary": summary,\
                "text_attention_mask": text_encoding["attention_mask"],\
                "labels": labels.flatten(),\
                "labels_attention_mask": summary_encoding["attention_mask"]}


class SummaryPLDataModule(pl.LightningDataModule):
    
    def __init__(\
            self,\
            df_train, \
            df_val, \
            tokenizer, \
            batch_size=2, \
            text_max_length=512, \
            summary_max_length=128 \
                ):
        super().__init__()
        
        self.df_train = df_train
        self.df_val = df_val
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.text_max_length = text_max_length
        self.summary_max_length = summary_max_length
        
    def setup(self, stage=None):
        
        self.train_dataset = SummaryDataset(\
                self.df_train,\
                self.tokenizer,\
                self.text_max_length,\
                self.summary_max_length \
                )
        self.val_dataset = SummaryDataset(\
                self.df_val,\
                self.tokenizer,\
                self.text_max_length,\
                self.summary_max_length \
                )
        
    def train_dataloader(self):
        
        return DataLoader(\
                self.train_dataset, \
                batch_size=self.batch_size, \
                shuffle=True, \
                num_workers=16 \
                )
    
    def val_dataloader(self):
        
        return DataLoader(\
                self.val_dataset, \
                batch_size=self.batch_size, \
                shuffle=False, \
                num_workers=16 \
                )

class Summarizer(pl.LightningModule):

    def __init__(self, lr=1e-5, model_name="t5-base"):

        super().__init__()
        self.model= T5ForConditionalGeneration.from_pretrained(\
                model_name,\
                return_dict=True \
                )

        self.lr = lr
        self.summary_max_length = 128
        self.text_max_length = 512
        self.number_beams = 1
        self.use_early_stopping = True

    def forward(self, input_ids, \
            attention_mask, decoder_attention_mask,\
            labels=None):

        my_output = self.model(input_ids, \
                attention_mask=attention_mask, \
                labels=labels, \
                decoder_attention_mask=decoder_attention_mask \
                )

        return my_output.loss, my_output.logits

    def training_step(self, batch, batch_index=None):

        input_ids = batch["input_ids"][:,0,...]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"][:,0,...]

        loss, outputs = self.forward(input_ids=input_ids,\
                attention_mask=attention_mask,\
                decoder_attention_mask=labels_attention_mask,\
                labels=labels\
                )

        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_index=None):

        with torch.no_grad():

            input_ids = batch["input_ids"][:,0,...]
            attention_mask = batch["text_attention_mask"]
            labels = batch["labels"]
            labels_attention_mask = batch["labels_attention_mask"][:,0,...]

            loss, outputs = self.forward(input_ids=input_ids,\
                    attention_mask=attention_mask,\
                    decoder_attention_mask=labels_attention_mask,\
                    labels=labels\
                    )


            self.log("validation_loss", loss)

        return loss

    def configure_optimizers(self):

        return torch.optim.Adam(\
                self.parameters(),\
                lr=self.lr)

    def summarize(self, text, tokenizer):

        with torch.no_grad():

            text_encoding = tokenizer(\
                    text,\
                    max_length=self.text_max_length,\
                    padding="max_length",\
                    return_attention_mask=True,\
                    add_special_tokens=True,\
                    return_tensors="pt",\
                    truncation=True\
                                            )
            input_ids = text_encoding["input_ids"]
            attention_mask = text_encoding["attention_mask"]


            summary_ids = self.model.generate(\
                    input_ids=input_ids,\
                    attention_mask=attention_mask,\
                    max_length=self.summary_max_length,\
                    num_beams=self.number_beams,\
                    repetition_penalty=3.0,\
                    length_penalty=1.0,\
                    early_stopping=self.use_early_stopping\
                               )

            decoded_list = [tokenizer.decode(id_elem,\
                    clean_up_tokenization_spaces=True,\
                    skip_special_tokens=True) for id_elem in summary_ids]

            decoded = "".join(decoded_list)

        return decoded

def summarize_train(df_train, df_val, number_epochs=1, \
        batch_size=1, model_name="t5-small"):

    # instantiate the tokenizer and model

    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)

    model = Summarizer(model_name=model_name)

    data_module = SummaryPLDataModule(df_train, \
            df_val,\
            tokenizer, \
            batch_size=batch_size \
            )

    trainer = pl.Trainer(max_epochs=number_epochs,\
        accelerator="gpu",\
        devices=1)

    trainer.fit(model, data_module)

    return model

def summarize_validate(model, df_val, number_samples=1):

    summary = ""

    for sample_number in range(number_samples):

        sample_index = np.random.randint(len(df_val))
        sample_text = df_val.iloc[sample_index]["text"]

        sample_summary = df_val.iloc[sample_index]["summary"]


        summarized = model.summarize(sample_text, tokenizer)
        msg = f"\nFull text: \n    {sample_text}"
        msg += f"\nSummary target: \n    {sample_summary}"

        msg += f"\nSummarized text: \n    {summarized}\n\n"

        summary += msg

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_name", type=str, default="t5-small",\
            help="model name, options: t5-small, t5-base")
    parser.add_argument("-e", "--number_epochs", type=int, default=1,\
            help="number of epochs to train")
    parser.add_argument("-b", "--batch_size", type=int, default=1,\
            help="number of samples per batch")
    parser.add_argument("-s", "--number_samples", type=int, default=1,\
            help="number of samples for summarization examples")


    args = parser.parse_args()
    model_name = args.model_name
    
    # load the csv data as pandas dataframes
    filepath = os.path.join("data", "cnn_dailymail", "train.csv")
    test_filepath = os.path.join("cnn_dailymail", "test.csv")
    val_filepath = os.path.join("cnn_dailymail", "validation.csv")
    df = pd.read_csv(filepath)#[:4000]
    df_val = pd.read_csv(val_filepath)#[:400]
    df_test = pd.read_csv(test_filepath)
    df_train = df[["article", "highlights"]]
    df_train.columns = ["text", "summary"]
    df_test = df_test[["article", "highlights"]]
    df_test.columns = ["text", "summary"]
    df_val = df_val[["article", "highlights"]]
    df_val.columns = ["text", "summary"]

    model = summarize_train(df_train, df_val, number_epochs=args.number_epochs,\
            batch_size=args.batch_size,\
            model_name=args.model_name)


    tokenizer = T5Tokenizer.from_pretrained(args.model_name, model_max_length=512)
    examples = summarize_validate(model, df_val, number_samples=args.number_samples)

    with open("examples.txt", "w") as f:
        f.write(examples)
    import pdb; pdb.set_trace()
    torch.save("model_temp.pt", model.state_dict())
