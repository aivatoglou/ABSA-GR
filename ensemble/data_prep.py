import re

import pandas as pd
from easynmt import EasyNMT


def machine_translation(train_data: pd.DataFrame, language: str) -> pd.DataFrame:

    model = EasyNMT("opus-mt", max_loaded_models=4)

    # Mask the aspect in order to NOT be translated
    sent = []
    for index, row in train_data.iterrows():
        index_ = row["text"].find(row["target"])
        size_ = len(row["target"])
        sent.append(row["text"][:index_] + "12345" + row["text"][index_ + size_ :])

    input_sentences = sent
    sentences = model.translate(
        input_sentences, source_lang="el", target_lang=language, batch_size=16
    )
    output_sentences = model.translate(
        sentences, source_lang=language, target_lang="el", batch_size=16
    )

    print(f"el-{language}-el done")

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
