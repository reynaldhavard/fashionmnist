from tqdm import tqdm
import torch
from fashionmnist.utils import Writer, ModelCheckpoint


def get_loss_fn():
    """
    Return the Cross Entropy Loss function from PyTorch
    """
    return torch.nn.CrossEntropyLoss()


def get_optimizer(model, learning_rate):
    """
    Initialize a Adam optimizer
    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_step(model, dataloader, loss_fn, optimizer, device):
    """
    Set the model to training mode, iterate through the dataloader, compute the
    loss and back-propagate for each batch, compute the accuracy.
    """
    model.train()

    train_loss, train_acc = 0, 0

    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        logits = model(X)

        loss = loss_fn(logits, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(logits, dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred_class)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def val_test_step(model, dataloader, loss_fn, device):
    """
    Set the model to eval mode, iterate through the dataloader in inference
    mode, get the loss and the accuracy.
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)

            logits = model(X)

            test_loss += loss_fn(logits, y).item()
            y_pred_class = torch.argmax(logits, dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred_class)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    epochs,
    device,
    save_model_path,
    writer_path,
    extra_info,
):
    """
    - Initialize a ModelCheckpoint instance and a Writer instance.
    - Train the model for epochs epochs.
    - Store the loss and the accuracy for the train and the validation sets in
    a dictionary and using the Writer so it can be visualized with Tensorboard.
    - Save the model with the lowest validation loss using the ModelCheckpoint.
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    model_checkpoint = ModelCheckpoint(
        save_model_path,
        model_name=model.__class__.__name__,
        extra_info=extra_info,
    )
    writer = Writer(
        writer_path, model_name=model.__class__.__name__, extra_info=extra_info
    )

    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device
        )
        val_loss, val_acc = val_test_step(
            model, val_dataloader, loss_fn, device
        )

        print(
            f"Epoch: {epoch} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        model_checkpoint.update(val_loss, model, epoch, optimizer)
        writer.update(train_loss, val_loss, train_acc, val_acc, epoch)

    writer.close()

    return results
