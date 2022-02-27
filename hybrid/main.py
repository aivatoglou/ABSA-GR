import sys

sys.path.append("../utils")

import random
import time
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torchvision
from data_prep import machine_translation
from modelsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer, logging
from utils import (
    ImbalancedDatasetSampler,
    language_model_preprocessing,
    translated_preprocessing,
)

from bertGRU import BERTGRUSentiment
from train_model import epoch_time, evaluate, train

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device for training: {device}")

logging.set_verbosity_error()

########## HYPER-PARAMETERS ##########
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.1
N_EPOCHS = 100
SEED = 0
use_sampling = True
classes = 3
######################################

# Control sources of randomness
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
logging.set_verbosity_error()

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
print(f"Samples removed: {len(ids_to_drop)}")

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

print(f"Train-set class balance: {Counter(train_data['sentiment'])}")
print(f"Val-set class balance: {Counter(val_data['sentiment'])}")
print(f"Test-set class balance: {Counter(test_data['sentiment'])}")

if use_sampling:

    m_0 = train_data[train_data["sentiment"] == 0]  # 1671 samples
    m_1 = train_data[train_data["sentiment"] == 1]  # 4720 samples
    m_2 = train_data[train_data["sentiment"] == 2]  # 752  samples

    m_2_fr = machine_translation(m_2, "mul", "en")
    m_2 = pd.concat([m_2, m_2_fr])

    m_2_fi = machine_translation(m_2, "el", "fr")
    m_2 = pd.concat([m_2, m_2_fi])

    m_0_fr = machine_translation(m_0, "mul", "en")
    m_0 = pd.concat([m_0, m_0_fr])

    train_data = pd.concat([m_0, m_1, m_2])

    # Free some resources from GPU
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

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
tokenizer._pad_token_type_id = 0

# Tokenize dataset
encoded_data_train = tokenizer(
    train_data["text"].tolist(),
    train_data["target"].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
    add_special_tokens=True,
    return_token_type_ids=True,
    return_attention_mask=True,
)

encoded_data_test = tokenizer(
    test_data["text"].tolist(),
    test_data["target"].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
    add_special_tokens=True,
    return_token_type_ids=True,
    return_attention_mask=True,
)

encoded_data_val = tokenizer(
    val_data["text"].tolist(),
    val_data["target"].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
    add_special_tokens=True,
    return_token_type_ids=True,
    return_attention_mask=True,
)

# Train-set
input_ids_train = encoded_data_train["input_ids"]
attention_masks_train = encoded_data_train["attention_mask"]
token_type_ids_train = encoded_data_train["token_type_ids"]
labels_train = torch.tensor(train_data["sentiment"].tolist())

# Test-set
input_ids_test = encoded_data_test["input_ids"]
attention_masks_test = encoded_data_test["attention_mask"]
token_type_ids_test = encoded_data_test["token_type_ids"]
labels_test = torch.tensor(test_data["sentiment"].tolist())

# Validation-set
input_ids_val = encoded_data_val["input_ids"]
attention_masks_val = encoded_data_val["attention_mask"]
token_type_ids_val = encoded_data_val["token_type_ids"]
labels_val = torch.tensor(val_data["sentiment"].tolist())

# Train dataloader
dataset_train = TensorDataset(
    input_ids_train, token_type_ids_train, attention_masks_train, labels_train
)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    sampler=ImbalancedDatasetSampler(dataset_train, labels_train),
    drop_last=True,
)

# Test dataloader
dataset_test = TensorDataset(
    input_ids_test, token_type_ids_test, attention_masks_test, labels_test
)

dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, drop_last=True)

# Validation dataloader
dataset_val = TensorDataset(
    input_ids_val, token_type_ids_val, attention_masks_val, labels_val
)

dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, drop_last=True)

# Initialize Greek-BERT
bert = BertModel.from_pretrained(
    "nlpaueb/bert-base-greek-uncased-v1", output_hidden_states=True
)
model = BERTGRUSentiment(bert, classes, BATCH_SIZE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

w = [list(labels_train).count(i) for i in range(classes)]
criterion = torch.nn.CrossEntropyLoss(
    weight=torch.FloatTensor([x / sum(w) for x in w]).cuda()
)

model = model.to(device)
criterion = criterion.to(device)
scaler = GradScaler()

# Freeze BERT layers
for name, param in model.named_parameters():
    if name.startswith("bert"):
        param.requires_grad = False

############ Model summary ############
# for batch in dataloader_train:
#    x = batch[0].to(device)
#    y = batch[1].to(device)
#    z = batch[2].to(device)
#    break

# print(summary(model, x, y, z, show_input=True))
#######################################

validation_threshold = np.inf
print("Train started...")
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(
        model, dataloader_train, optimizer, criterion, scaler, device
    )
    valid_loss, valid_acc = evaluate(
        model, dataloader_val, criterion, device, print_report=False
    )

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train f1: {train_acc*100:.2f}%")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. f1: {valid_acc*100:.2f}%")

    for g in optimizer.param_groups:
        g["lr"] = g["lr"] * 0.75
        print("Learning rate: ", g["lr"])

    if valid_loss < validation_threshold:
        validation_threshold = valid_loss
    else:
        break

# Calculate test loss and accuracy
test_loss, test_acc = evaluate(
    model, dataloader_test, criterion, device, print_report=True
)
print(f"Test Loss: {test_loss:.3f} | Test f1: {test_acc*100:.2f}%")
