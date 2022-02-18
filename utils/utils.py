import re
import unicodedata
from typing import Callable

import pandas as pd
import torch


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self,
        dataset,
        labels_train,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        self.indices = list(range(len(dataset))) if indices is None else indices

        self.callback_get_label = callback_get_label

        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = labels_train
        df.index = self.indices
        df = df.sort_index()
        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


def language_model_preprocessing(text):

    # Remove url
    text = re.sub(r"(\w+:\/\/\S+)", " ", text)
    # Remove dates and time
    text = re.sub(r"[0-9]{2}[\/,:][0-9]{2}[\/,:][0-9]{2,4}", " ", text)
    # Remove emails
    text = re.sub(r"(\w+@\w+.[\w+]{2,4})", " ", text)
    # Remove percentages
    text = re.sub(r"\d+(\%|\s\bpercent\b)", " ", text)
    # Remove special symbols
    text = re.sub(r"#|@|:|-", " ", text)

    if text.startswith("RT") | text.startswith("rt"):
        text = text[len("RT") :]

    # Multiline to single line codes
    text = " ".join(text.splitlines())

    # Remove extra spaces
    text = re.sub(" +", " ", text)
    # Strip string
    text = text.strip()

    # Needed in order to find if aspect is in text
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    return text


def translated_preprocessing(text):

    # Remove digits
    text = re.sub(r"[~^0-9]", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove duplicated words in row
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)

    # Remove url
    text = re.sub(r"(\w+:\/\/\S+)", " ", text)
    # Remove dates and time
    text = re.sub(r"[0-9]{2}[\/,:][0-9]{2}[\/,:][0-9]{2,4}", " ", text)
    # Remove emails
    text = re.sub(r"(\w+@\w+.[\w+]{2,4})", " ", text)
    # Remove percentages
    text = re.sub(r"\d+(\%|\s\bpercent\b)", " ", text)
    # Remove special symbols
    text = re.sub(r"#|@|:|-", " ", text)

    if text.startswith("RT") | text.startswith("rt"):
        text = text[len("RT") :]

    # Multiline to single line codes
    text = " ".join(text.splitlines())

    # Remove extra spaces
    text = re.sub(" +", " ", text)
    # Strip string
    text = text.strip()

    # Needed in order to find if aspect is in text
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    return text
