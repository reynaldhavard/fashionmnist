from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def get_predictions(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.inference_mode():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            y_pred_class = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_pred_class.cpu().numpy())

    return y_true, y_pred


def get_incorrect_predictions(model, dataloader, device, n_predictions=None):
    if n_predictions is None:
        n_predictions = len(dataloader)

    list_X = []
    list_y_true = []
    list_y_pred = []
    count = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            y_pred_class = torch.argmax(logits, dim=1)
            if y_pred_class != y:
                list_X.extend(X.cpu().numpy())
                list_y_true.extend(y.cpu().numpy())
                list_y_pred.extend(y_pred_class.cpu().numpy())
                count += 1
                if count == n_predictions:
                    break

    return list_X, list_y_true, list_y_pred


def plot_incorrect_predictions(
    model, dataloader, class_names, device, n_predictions=None
):
    if n_predictions is None:
        n_predictions = len(dataloader)
    n_predictions = int(np.sqrt(n_predictions)) ** 2

    list_X, list_y_true, list_y_pred = get_incorrect_predictions(
        model, dataloader, device, n_predictions
    )

    n_row = int(np.sqrt(n_predictions))
    fig, axes = plt.subplots(n_row, n_row, figsize=(12, 12))
    for i, (X, y_true, y_pred) in enumerate(
        zip(list_X, list_y_true, list_y_pred)
    ):
        axes[i // n_row, i % n_row].imshow(X.squeeze(), cmap="gray")
        axes[i // n_row, i % n_row].set_axis_off()
        axes[i // n_row, i % n_row].set_title(
            f"Truth: {class_names[y_true]}\nPredicted: {class_names[y_pred]}",
            fontdict={"fontsize": 10},
        )
    plt.show()


def get_confusion_matrix(y_true, y_pred, class_names):
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix) * 10,
        index=class_names,
        columns=class_names,
    )

    return df_cm


def plot_confusion_matrix(y_true, y_pred, class_names):
    df_cm = get_confusion_matrix(y_true, y_pred, class_names)

    plt.figure(figsize=(10, 5))
    sns.heatmap(df_cm, annot=True)
    plt.xticks(rotation=45)
    plt.show()


def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
