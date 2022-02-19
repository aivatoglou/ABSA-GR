import re

import pandas as pd
from easynmt import EasyNMT


def machine_translation(
    train_data: pd.DataFrame, language_model_1: str, language_model_2: str
) -> pd.DataFrame:

    """Data augmentation through machine translation.

    Keyword arguments:
    train_data -- a pandas dataframe containing the text and the target (brand).
    language_model_1 -- The source language for the translation (el or mul for multi-lingual models)
    language_model_2 -- The target language for the translation.

    Returns a new dataframe with the original and the augmented data.
    """

    model = EasyNMT("opus-mt", max_loaded_models=5)

    # Mask the aspect in order to NOT be translated
    sent = []
    for index, row in train_data.iterrows():
        index_ = row["text"].find(row["target"])
        size_ = len(row["target"])
        sent.append(row["text"][:index_] + "12345" + row["text"][index_ + size_ :])

    input_sentences = sent
    sentences = model.translate(
        input_sentences,
        source_lang=language_model_1,
        target_lang=language_model_2,
        batch_size=16,
    )
    output_sentences = model.translate(
        sentences, source_lang=language_model_2, target_lang="el", batch_size=16
    )

    print(f"Translation done!")

    extra_aspects = train_data["target"].tolist()
    extra_sentiments = train_data["sentiment"].tolist()

    # Put the aspect back in the sentence
    extra_data = []
    for x, item in enumerate(output_sentences):

        # Remove repetitive characters (3+) as a result of the translator
        item = re.sub(r"(\w)\1(\1+)", r"\1", item)
        item = item.strip()

        index_ = item.find("12345")
        size_ = len("12345")
        extra_data.append(item[:index_] + extra_aspects[x] + item[index_ + size_ :])

    extra_dataset = pd.DataFrame(
        list(zip(extra_data, extra_aspects, extra_sentiments)),
        columns=["text", "target", "sentiment"],
    )

    return extra_dataset
