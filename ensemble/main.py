import random
import sys

sys.path.append("../utils")

from collections import Counter

import numpy as np
import pandas as pd
import torch
from data_prep import machine_translation
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AdamW,
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging,
)
from utils import language_model_preprocessing, translated_preprocessing

from train import (
    accuracy_per_class,
    evaluate_model,
    f1_score_func,
    test_model,
    train_model,
)

########## HYPER-PARAMETERS ##########
SEED = 0
EPOCHS = 5
LEARNING_RATE = 2e-5
language_model_1 = "nlpaueb/bert-base-greek-uncased-v1"
language_model_2 = "xlm-roberta-base"
seq_length = 256
BATCH_SIZE = 8
use_sampling = True
classes = 3
######################################

# Control sources of randomness
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
logging.set_verbosity_error()

# Training device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device for training: {device}")

# Load dataset
dataset = pd.read_csv("../data/dataset.csv", index_col=False, sep="\t")

# Map labels - convert to 3-classes problem
labels_dict = {"-2": 0, "-1": 0, "0": 1, "1": 2, "2": 2}
dataset["sentiment"] = dataset["sentiment"].astype(int).astype(str)
dataset["sentiment"] = dataset["sentiment"].map(labels_dict)
dataset = dataset.reset_index(drop=True)

# Dataset Preprocessing
dataset["text"] = dataset["text"].apply(language_model_preprocessing)
dataset["target"] = dataset["target"].apply(language_model_preprocessing)

# Discard rows where aspect is not in text
ids_to_drop = []
for index, row in dataset.iterrows():
    if row["target"] not in row["text"]:
        ids_to_drop.append(index)

dataset = dataset[~dataset.index.isin(ids_to_drop)]
dataset = dataset.reset_index(drop=True)

# Shuffle dataset
dataset = shuffle(dataset, random_state=SEED)

# Train-test split
train_data, test_data = train_test_split(
    dataset, test_size=0.2, random_state=SEED, stratify=dataset["sentiment"].values
)

# Validation set
test_data, val_data = train_test_split(
    test_data, train_size=0.5, random_state=SEED, stratify=test_data["sentiment"].values
)

print(f"Initial train-set class balance: {Counter(train_data['sentiment'])}")
print(f"Val-set class balance: {Counter(val_data['sentiment'])}")
print(f"Test-set class balance: {Counter(test_data['sentiment'])}")

if use_sampling:

    m_0 = train_data[train_data["sentiment"] == 0]  # 1671
    m_1 = train_data[train_data["sentiment"] == 1]  # 4720
    m_2 = train_data[train_data["sentiment"] == 2]  # 752

    m_2_fr = machine_translation(m_2, "mul", "en")
    m_2 = pd.concat([m_2, m_2_fr])

    m_2_fi = machine_translation(m_2, "el", "fr")
    m_2 = pd.concat([m_2, m_2_fi])

    m_0_fr = machine_translation(m_0, "mul", "en")
    m_0 = pd.concat([m_0, m_0_fr])

    train_data = pd.concat([m_0, m_1, m_2])

    del m_0
    del m_1
    del m_2
    torch.cuda.empty_cache()

    train_data["text"] = train_data["text"].apply(translated_preprocessing)
    train_data["target"] = train_data["target"].apply(translated_preprocessing)

    ids_to_drop = []
    for index, row in train_data.iterrows():
        if row["target"] not in row["text"]:
            ids_to_drop.append(index)

    train_data = train_data[~train_data.index.isin(ids_to_drop)]
    train_data = train_data.reset_index(drop=True)
    print(f"Samples removed: {len(ids_to_drop)}")

    train_data = shuffle(train_data, random_state=SEED)
    print(f"Train set class balance after sampling: {Counter(train_data['sentiment'])}")

########## Greek-BERT initialization ##########

tokenizer_bert = AutoTokenizer.from_pretrained(language_model_1)
tokenizer_bert._pad_token_type_id = 0
aux_sentence = "target"

# Tokenize train and test sets
encoded_data_train = tokenizer_bert(
    train_data["text"].tolist(),
    train_data[aux_sentence].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=seq_length,
    add_special_tokens=True,
    return_token_type_ids=True,
    return_attention_mask=True,
)

encoded_data_test = tokenizer_bert(
    test_data["text"].tolist(),
    test_data[aux_sentence].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=seq_length,
    add_special_tokens=True,
    return_token_type_ids=True,
    return_attention_mask=True,
)

encoded_data_val = tokenizer_bert(
    val_data["text"].tolist(),
    val_data[aux_sentence].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=seq_length,
    add_special_tokens=True,
    return_token_type_ids=True,
    return_attention_mask=True,
)

labels = "sentiment"

# Train-set
input_ids_train = encoded_data_train["input_ids"]
attention_masks_train = encoded_data_train["attention_mask"]
token_type_ids_train = encoded_data_train["token_type_ids"]
labels_train = torch.tensor(train_data[labels].tolist())

# Test-set
input_ids_test = encoded_data_test["input_ids"]
attention_masks_test = encoded_data_test["attention_mask"]
token_type_ids_test = encoded_data_test["token_type_ids"]
labels_test = torch.tensor(test_data[labels].tolist())

# Validation-set
input_ids_val = encoded_data_val["input_ids"]
attention_masks_val = encoded_data_val["attention_mask"]
token_type_ids_val = encoded_data_val["token_type_ids"]
labels_val = torch.tensor(val_data[labels].tolist())

num_labels = len(dataset["sentiment"].unique())

classifier_1 = BertForSequenceClassification.from_pretrained(
    language_model_1,
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False,
)
classifier_1.to(device)

# Train dataloader
dataset_train = TensorDataset(
    input_ids_train, token_type_ids_train, attention_masks_train, labels_train
)

dataloader_train_bert = DataLoader(dataset_train, sampler=None, batch_size=BATCH_SIZE)

# Test dataloader
dataset_test = TensorDataset(
    input_ids_test, token_type_ids_test, attention_masks_test, labels_test
)

dataloader_test_bert = DataLoader(dataset_test, sampler=None, batch_size=BATCH_SIZE)

# Validation dataloader
dataset_val = TensorDataset(
    input_ids_val, token_type_ids_val, attention_masks_val, labels_val
)

dataloader_val_bert = DataLoader(dataset_val, sampler=None, batch_size=BATCH_SIZE)

optimizer_bert = AdamW(
    classifier_1.parameters(), lr=LEARNING_RATE, eps=1e-8, weight_decay=0.1
)
scheduler_bert = get_linear_schedule_with_warmup(
    optimizer_bert,
    num_warmup_steps=0.1 * len(dataloader_train_bert) * EPOCHS,
    num_training_steps=len(dataloader_train_bert) * EPOCHS,
)

########## XML-RoBERTa initialiaztion ##########

tokenizer_roberta = AutoTokenizer.from_pretrained(language_model_2)
tokenizer_roberta._pad_token_type_id = 0
aux_sentence = "target"

# Tokenize train and test sets
encoded_data_train = tokenizer_roberta(
    train_data["text"].tolist(),
    train_data[aux_sentence].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=seq_length,
    add_special_tokens=True,
    return_token_type_ids=True,
    return_attention_mask=True,
)

encoded_data_test = tokenizer_roberta(
    test_data["text"].tolist(),
    test_data[aux_sentence].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=seq_length,
    add_special_tokens=True,
    return_token_type_ids=True,
    return_attention_mask=True,
)

encoded_data_val = tokenizer_roberta(
    val_data["text"].tolist(),
    val_data[aux_sentence].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=seq_length,
    add_special_tokens=True,
    return_token_type_ids=True,
    return_attention_mask=True,
)

labels = "sentiment"

# Train-set
input_ids_train = encoded_data_train["input_ids"]
attention_masks_train = encoded_data_train["attention_mask"]
token_type_ids_train = encoded_data_train["token_type_ids"]
labels_train = torch.tensor(train_data[labels].tolist())

# Test-set
input_ids_test = encoded_data_test["input_ids"]
attention_masks_test = encoded_data_test["attention_mask"]
token_type_ids_test = encoded_data_test["token_type_ids"]
labels_test = torch.tensor(test_data[labels].tolist())

# Validation-set
input_ids_val = encoded_data_val["input_ids"]
attention_masks_val = encoded_data_val["attention_mask"]
token_type_ids_val = encoded_data_val["token_type_ids"]
labels_val = torch.tensor(val_data[labels].tolist())

num_labels = len(dataset["sentiment"].unique())
classifier_2 = BertForSequenceClassification.from_pretrained(
    language_model_2,
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False,
)

classifier_2.to(device)

# Freeze BERT layers
for name, param in classifier_1.named_parameters():
    if name.startswith("bert"):
        param.requires_grad = False

# Freeze RoBERTa layers
for name, param in classifier_2.named_parameters():
    if name.startswith("bert"):
        param.requires_grad = False

# Train dataloader
dataset_train = TensorDataset(
    input_ids_train, token_type_ids_train, attention_masks_train, labels_train
)

dataloader_train_roberta = DataLoader(
    dataset_train, sampler=None, batch_size=BATCH_SIZE
)

# Test dataloader
dataset_test = TensorDataset(
    input_ids_test, token_type_ids_test, attention_masks_test, labels_test
)

dataloader_test_roberta = DataLoader(dataset_test, sampler=None, batch_size=BATCH_SIZE)

# Validation dataloader
dataset_val = TensorDataset(
    input_ids_val, token_type_ids_val, attention_masks_val, labels_val
)

dataloader_val_roberta = DataLoader(dataset_val, sampler=None, batch_size=BATCH_SIZE)

optimizer_roberta = AdamW(
    classifier_2.parameters(), lr=LEARNING_RATE, eps=1e-8, weight_decay=0.1
)
scheduler_roberta = get_linear_schedule_with_warmup(
    optimizer_roberta,
    num_warmup_steps=0.1 * len(dataloader_train_roberta) * EPOCHS,
    num_training_steps=len(dataloader_train_roberta) * EPOCHS,
)

print(f"Training phase ...")
for epoch in range(1, EPOCHS + 1):

    train_loss_bert = train_model(
        classifier_1, dataloader_train_bert, optimizer_bert, scheduler_bert, device
    )
    train_loss_roberta = train_model(
        classifier_2,
        dataloader_train_roberta,
        optimizer_roberta,
        scheduler_roberta,
        device,
    )

    val_loss_bert, predictions_bert, true_vals = evaluate_model(
        dataloader_val_bert, classifier_1, device
    )
    val_loss_roberta, predictions_roberta, true_vals = evaluate_model(
        dataloader_val_roberta, classifier_2, device
    )

    # average the logits of the two classifiers
    ensemble_predictions = (predictions_bert + predictions_roberta) / 2.0
    val_f1 = f1_score_func(ensemble_predictions, true_vals)
    print(
        f"Epoch #{epoch} - bert_t_loss {train_loss_bert} - bert_v_loss {val_loss_bert} - roberta_t_loss {train_loss_roberta} - roberta_v_loss {val_loss_roberta} mean_v_f1 {val_f1}"
    )

predictions_bert, true_vals = test_model(dataloader_test_bert, classifier_1, device)
predictions_roberta, true_vals = test_model(
    dataloader_test_roberta, classifier_2, device
)
ensemble_predictions = (predictions_bert + predictions_roberta) / 2.0
print(accuracy_per_class(ensemble_predictions, true_vals))
