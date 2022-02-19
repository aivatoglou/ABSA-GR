import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def train_model(bert_model, dataloader_train, optimizer, scheduler, device):

    """The architecture's training routine."""

    bert_model.train()
    loss_train_total = 0

    for batch_idx, batch in enumerate(dataloader_train):

        # set gradient to 0
        bert_model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "labels": batch[3],
        }

        loss, _ = bert_model(
            inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            return_dict=False,
        )

        # Compute train loss
        loss_train_total += loss.item()

        loss.backward()

        # gradient accumulation
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # torch.save(bert_model.state_dict(), f"models/ BERT_ft_epoch{epoch}.model")
    loss_train_avg = loss_train_total / len(dataloader_train)
    return loss_train_avg


def evaluate_model(dataloader_val, bert_model, device):

    """The architecture's evaluation routine."""

    # evaluation mode
    bert_model.eval()

    # tracking variables
    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        # load into GPU
        batch = tuple(b.to(device) for b in batch)

        # define inputs
        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "labels": batch[3],
        }

        # compute logits
        with torch.no_grad():

            loss, logits = bert_model(
                inputs["input_ids"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                return_dict=False,
            )

        # Compute validation loss
        loss_val_total += loss.item()

        # compute accuracy
        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    # compute average loss
    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


def test_model(dataloader_test, bert_model, device):

    """The architecture's test routine."""

    bert_model.eval()
    predictions, true_vals = [], []

    for batch in dataloader_test:

        # load into GPU
        batch = tuple(b.to(device) for b in batch)

        # define inputs
        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "labels": batch[3],
        }

        # compute logitsq
        with torch.no_grad():

            _, logits = bert_model(
                inputs["input_ids"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                return_dict=False,
            )

        # compute accuracy
        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return predictions, true_vals


def f1_score_func(preds, labels):

    """Calculates the macro F1-score."""

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average="macro")


def accuracy_per_class(preds, labels):

    """Calculates the accuracy per class."""

    # make prediction
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    print(confusion_matrix(labels_flat, preds_flat))
    print(classification_report(labels_flat, preds_flat))

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f"Class: {label}")
        print(f"Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n")
