import matplotlib.pyplot as plt
import torch
import os
from os.path import exists

def draw_training_curves(train_losses, test_losses, epochs, output_dir):
    """
    plots training and testing loss/accuracy curves
    params: train_losses = training loss
            test_losses = validation loss
            curve_name = loss or accuracy
    """
    plt.clf()
    plt.xlim([0, epochs])
    plt.plot(train_losses, label="Training {}".format("LOSS"))
    plt.plot(test_losses, label="Testing {}".format("loss"))
    plt.legend(frameon=False)
    plt.savefig(output_dir + "/training_curves.png")

def save_checkpoint(model, epoch, output_dir, final_flag):
    if not exists(output_dir):
        try:
            os.mkdir(output_dir)
        except Exception as e:
            print("EXIT: Unable to create outpu dir: {}".format(e))
            exit()
    fname = "unet_checkpoint.pkl"
    state = {"epoch": epoch, "state_dict": model.state_dict()}
    if final_flag:
        fname = "final_" + fname
    torch.save(state, output_dir + "/" + fname)
    return


def load_checkpoint(checkpoint_path, model):
    epoch = 0
    try:
        state = torch.load(checkpoint_path)
        model.load_state_dict(state["state_dict"])
        epoch = state["epoch"]
    except Exception as e:
        print(
            "WARNING: unable to open checkpoint {}. This is OK if you train from scratch".format(
                e
            )
        )
    return model, epoch
