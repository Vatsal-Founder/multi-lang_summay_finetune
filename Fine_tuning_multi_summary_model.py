# %%
import os, numpy as np, random
from datasets import load_dataset, interleave_datasets

from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          EarlyStoppingCallback)

import torch

# %%
from transformers import pipeline, set_seed
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
#from datasets import load_dataset, load_metric

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import nltk
from nltk.tokenize import sent_tokenize

from tqdm import tqdm
import torch

nltk.download("punkt")

# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
model = "csebuetnlp/mT5_multilingual_XLSum"

tokenizer = AutoTokenizer.from_pretrained(model)  #load a tokenizer

model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

# %%
#dowload & unzip data

!wget https://github.com/krishnaik06/datasets/raw/refs/heads/main/summarizer-data.zip
!unzip summarizer-data.zip

# %%
dataset_samsum = load_from_disk('samsum_dataset')
dataset_samsum

# %%
split_lengths = [len(dataset_samsum[split])for split in dataset_samsum]

print(f"Split lengths: {split_lengths}")
print(f"Features: {dataset_samsum['train'].column_names}")
print("\nDialogue:")

print(dataset_samsum["test"][2]["dialogue"])

print("\nSummary:")

print(dataset_samsum["test"][2]["summary"])

# %%
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['dialogue'] , max_length = 1024, truncation = True )

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['summary'], max_length = 128, truncation = True )

    return {
        'input_ids' : input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }


# %%
dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched = True)

# %%
dataset_samsum_pt['test']

# %%
# Training

from transformers import DataCollatorForSeq2Seq

seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

# %%
from transformers import TrainingArguments, Trainer


trainer_args = TrainingArguments(
            output_dir="model_mtf-final",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,     # effective batch = 8
            eval_strategy="steps",       # <-- correct arg name
            eval_steps=500,
            save_steps=1000000,                # int, not 1e6 float
            logging_steps=10,
            warmup_steps=200,                  # lower warmup to cut steps/memory
            weight_decay=0.01,
            optim="adafactor",                 # big memory win vs AdamW
            dataloader_pin_memory=False,       # CUDA-specific; off on MPS
            report_to="none",
            load_best_model_at_end=False,      # can set True if you also set metric_for_best_model
        )

trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["test"],
                  eval_dataset=dataset_samsum_pt["validation"])

# %%
trainer.train()

# %%
# Evaluation
### lst[1,2,3,4,5,6]-> [1,2,3][4,5,6]
def generate_batch_sized_chunks(list_of_elements, batch_size):
    """split the dataset into smaller batches that we can process simultaneously
    Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]



def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                               batch_size=16, device=device,
                               column_text="article",
                               column_summary="highlights"):
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches), total=len(article_batches)):

        inputs = tokenizer(article_batch, max_length=1024,  truncation=True,
                        padding="max_length", return_tensors="pt")

        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                         attention_mask=inputs["attention_mask"].to(device),
                         length_penalty=0.8, num_beams=8, max_length=128)
        ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''

        # Finally, we decode the generated texts,
        # replace the  token, and add the decoded texts with the references to the metric.
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
               for s in summaries]

        decoded_summaries = [d.replace("", " ") for d in decoded_summaries]


        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    #  Finally compute and return the ROUGE scores.
    score = metric.compute()
    return score


# %%
!pip install evaluate

# %%
import evaluate

rouge_metric = evaluate.load('rouge')
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
#rouge_metric = load_metric('rouge')

# %%
score = calculate_metric_on_test_ds(
    dataset_samsum['test'][0:10], rouge_metric, trainer.model, tokenizer, batch_size = 2, column_text = 'dialogue', column_summary= 'summary'
)

# Directly use the scores without accessing fmeasure or mid
rouge_dict = {rn: score[rn] for rn in rouge_names}

# Convert the dictionary to a DataFrame for easy visualization
import pandas as pd
pd.DataFrame(rouge_dict, index=[f'pegasus'])

# %%
## Save model
model_pegasus.save_pretrained("mt5-model")
## Save tokenizer
tokenizer.save_pretrained("tokenizer")

# %%



