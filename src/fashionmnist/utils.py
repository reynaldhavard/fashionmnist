import os
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def get_extended_model_name(model_name, extra_info):
    if extra_info is None:
        name = f"{model_name}.pth"
    else:
        name = model_name
        for key, value in extra_info.items():
            name = name + f"_{key}={value}"
        name = name + ".pth"

    return name


class Writer:
    def __init__(self, dir_path, model_name, extra_info):
        self.log_dir = os.path.join(
            dir_path, get_extended_model_name(model_name, extra_info)
        )
        print(f"Created SummaryWriter, saving to {self.log_dir}")
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def update(self, train_loss, val_loss, train_acc, val_acc, epoch):
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/val", val_loss, epoch)
        self.writer.add_scalar("accuracy/train", train_acc, epoch)
        self.writer.add_scalar("accuracy/val", val_acc, epoch)

    def close(self):
        self.writer.close()


class ModelCheckpoint:
    def __init__(self, dir_path, model_name, extra_info):
        self.min_loss = None
        self.path = os.path.join(
            dir_path, get_extended_model_name(model_name, extra_info)
        )
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def update(self, loss, model, epoch, optimizer):
        if self.min_loss is None or self.min_loss > loss:
            print(f"Updated best model at {self.path}")
            self.min_loss = loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "hidden_units": model.get_hidden_units(),
                    "model_name": model.__class__.__name__,
                },
                self.path,
            )


def load_checkpoint(model_path):
    checkpoint = torch.load(model_path)

    return checkpoint


def plot_training_curves(results):
    epochs = list(range(len(results["train_loss"])))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].plot(epochs, results["train_loss"], label="train_loss")
    axes[0].plot(epochs, results["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()

    axes[1].plot(epochs, results["train_acc"], label="train_acc")
    axes[1].plot(epochs, results["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].legend()
