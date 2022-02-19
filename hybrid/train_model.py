import time

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.cuda.amp import autocast


def epoch_time(start_time, end_time):

    """Calculates the elapsed time between the epochs."""

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def class_report(preds, y):

    """Prints the classification report for the test set."""

    flat_preds = [x for sublist in preds for x in sublist]
    flat_truth = [x for sublist in y for x in sublist]

    print(classification_report(flat_truth, flat_preds))
    return


# Define binary_accuracy function
def binary_accuracy(preds, y):

    """Calculates the macro F1-score."""

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = y.flatten()

    return f1_score(labels_flat, preds_flat, average="macro")


# Define train function
def train(model, iterator, optimizer, criterion, scaler, device):

    """The architecture's training routine."""

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        batch = tuple(b.to(device) for b in batch)

        optimizer.zero_grad()

        with autocast():

            predictions = model(batch[0], batch[1], batch[2]).squeeze(1)

            loss = criterion(predictions, batch[3].to(torch.int64))

            acc = binary_accuracy(
                predictions.detach().cpu().numpy(), batch[3].cpu().numpy()
            )

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        scaler.update()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device, print_report):

    """The architecture's evaluation routine."""

    preds = []
    truth = []

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            batch = tuple(b.to(device) for b in batch)

            with autocast():

                predictions = model(batch[0], batch[1], batch[2]).squeeze(1)

                loss = criterion(predictions, batch[3].to(torch.int64))

                acc = binary_accuracy(
                    predictions.detach().cpu().numpy(), batch[3].cpu().numpy()
                )

                preds_flat = np.argmax(
                    predictions.detach().cpu().numpy(), axis=1
                ).flatten()
                preds.append(preds_flat)
                truth.append(batch[3].cpu().numpy())

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    if print_report:
        print(class_report(preds, truth))

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
