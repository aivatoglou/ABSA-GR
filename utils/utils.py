import re
import unicodedata
from typing import Callable, List

import pandas as pd
import torch


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    """Class that handles the imbalance problem.
    Acts as the sampler for the PyTorch dataloaders.

    Keyword arguments:
    dataset -- a pandas series containing the text.
    labels_train: a list containing the labels.

    Returns the indices used in the PyTorch dataloader.
    """

    def __init__(
        self,
        dataset: pd.Series,
        labels_train: list,
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


def language_model_preprocessing(text: pd.Series) -> pd.Series:

    """Text preprocessing function.

    Keyword arguments:
    text -- a pandas series containing the text.

    Returns a new pd.Series with the pre-processed data.
    """

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


def translated_preprocessing(text: pd.Series) -> pd.Series:

    """Text preprocessing function (applied after machine translation).

    Keyword arguments:
    text -- a pandas series containing the text.

    Returns a new pd.Series with the pre-processed data.
    """

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
