import glob, os
import re, argparse
import random, time
from os.path import exists
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# scikit-image library fro CV
from skimage import io, transform
from skimage.color import gray2rgb, rgb2gray,rgba2rgb
import skimage.filters
# all PyTorch imports
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from PIL import Image as im
from pathlib import Path
from bin.readimage import readimage
from bin.CellDataset import CellData
from bin.utils import draw_training_curves, save_checkpoint, load_checkpoint
from IPython import embed
#from tkinter import filedialog as fd

random.seed(10)
torch.manual_seed(10)

aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=2,                 # define number of output labels
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# for reproducibility (split of files into train and val)
random.seed(10)

def get_arguments():
    """
    Read command line arguments
    """
    parser = argparse.ArgumentParser(description=" Cell Segmentation Training ")
    parser.add_argument(
        "--train",
        action='store_true',
        help="execute script in training mode, if False inference is run based on checkpoint",
    )
    parser.add_argument(
        "--validation",
        action='store_true',
        help="execute script in validation mode, if False inference is run based on checkpoint",
    )
    parser.add_argument(
        "--inference",
        action='store_true',
        help="execute script in inference mode, and label unlabled data"
    )

    parser.add_argument(
        "--epochs", type=int, default=80, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="size of data batches"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="loss rate")
    args = parser.parse_args()
    return args

def create_dataloaders(rel_datapath, batch_size):
    """
    data loader
    """
    full_data = CellData(rel_datapath)
    train_size = int(0.8 * len(full_data))
    test_size  = len(full_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, val_loader



def train_model(model, epochs, train_loader, val_loader, lr, output_dir):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    train_loss = []
    validation_loss = []

    for epoch in tqdm(range(epochs)):
        model.train()
        for data_batch in train_loader:
            cell_img, mask_img, _ = data_batch
            cell_img  = cell_img.to(DEVICE)
            true_mask  = mask_img.to(DEVICE)

            optimizer.zero_grad()
            pred_mask, label= model(cell_img)
            loss = criterion(pred_mask, true_mask)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        model.eval()
        for data_batch in val_loader:
            cell_img, mask_img,_ = data_batch
            cell_img  = cell_img.to(DEVICE)
            true_mask = mask_img.to(DEVICE)
            with torch.no_grad():
                pred_mask,label = model(cell_img)
                loss      = criterion(pred_mask, true_mask)
                validation_loss.append(loss.item())
        # every 2 epochs we save the data to a checkpoint
        if epoch % 2 ==0:
            save_checkpoint(model, epoch+1, output_dir, 0)
    save_checkpoint(model, epochs, output_dir, 1)

    return model,epochs,train_loss, validation_loss




def train_seg_model(rel_datapath, output_dir,checkpoint_path, epochs, args):

    batch_size = args.batch_size
    lr         = args.lr

    train_loader, val_loader = create_dataloaders(rel_datapath, batch_size)
    model = smp.Unet('resnet34', classes=1, aux_params=aux_params )
    executed_epochs = 0

    try:
        model, executed_epochs = load_checkpoint(checkpoint_path, model)
    except Exception as e:
        print('No valid checkpoint has been found {}'. format(e))
    model  = model.to(DEVICE)
    epochs = epochs - executed_epochs
    print("Executed_epochs {}".format(executed_epochs))

    if epochs == 0:
        print('The model has been already trained for provided number of epochs. Nothing to do!')
        exit()

    model, epochs, train_loss, val_loss = train_model(model, epochs, train_loader, val_loader, lr, output_dir)

    return model, epochs, train_loss, val_loss



def save_inference_results(pred_mask,true_mask, img_org,output_dir, i):

    img = pred_mask.detach().cpu().numpy().squeeze()
    img = (img *255/np.max(img)).astype('uint8')
    imgF = im.fromarray(img)
    imgF = imgF.resize((512, 512))
    imgF.save(output_dir + "/{}_pred_mask.png".format(i))


    img = true_mask.detach().cpu().numpy().squeeze()
    img = (img *255/np.max(img)).astype('uint8')
    imgF = im.fromarray(img)
    imgF = imgF.resize((512, 512))
    imgF.save(output_dir + "/{}_true_mask.png".format(i))

    img = img_org.detach().cpu().numpy().squeeze()
    imgF = im.fromarray(img)
    imgF = imgF.resize((512, 512))
    imgF.save(output_dir + "/{}_org_img.png".format(i))

    return


def run_inference_on_validation(rel_datapath, output_dir, model):

    train_loader, val_loader = create_dataloaders(rel_datapath, 1)

    model.eval()
    i = 0
    sig = nn.Sigmoid()

    for data_batch in val_loader:
        cell_img, mask_img, img_org = data_batch
        cell_img   = cell_img.to(DEVICE)
        true_mask  = mask_img.to(DEVICE)
        with torch.no_grad():
            pred_mask,label = model(cell_img)
            sig_pred_mask = sig(pred_mask)
            sig_pred_mask[sig_pred_mask >= 0.5]= 1
            sig_pred_mask[sig_pred_mask < 0.5] = 0
        save_inference_results(sig_pred_mask, mask_img, img_org,output_dir, i)
        i = i+1

    return

def predict_mask(original, model):
    """
    Given an original image return the mask
    """
    original = readimage(original)
    original = torch.unsqueeze(original, 0)
    original = original.to(DEVICE)
    model.eval()
    sig = nn.Sigmoid()
    mask = None
    with torch.no_grad():
        pred_mask,label = model(original)
        sig_pred_mask = sig(pred_mask)
        sig_pred_mask[sig_pred_mask >= 0.5]= 1
        sig_pred_mask[sig_pred_mask < 0.5] = 0
        img = sig_pred_mask.detach().cpu().numpy().squeeze()
        img = (img * 255 / np.max(img)).astype("uint8")
        mask = im.fromarray(img)
        # scale the 256 x 256 image up
        mask= mask.resize((512, 512))
    return mask

def run_inference(unlabeled_dir, model):
    """
    Processes unlabeled data and produces an image
    """
    #images = list(Path('./unlabeled').glob('*.png'))
    images = list(Path(unlabeled_dir).glob('**/*.png'))
    for original in tqdm(images):
        #originals
        original_image  = plt.imread(original)
        # embed()
        # Check if output directory exists already
        out_dir = unlabeled_dir + "/labeled"
        print(out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        img_name = out_dir + "/" + str(original.name).strip(".png")
        # plt.imsave(img_name +'.png', original_image, cmap='gray')
        # predicted_mask
        mask = predict_mask(io.imread(original), model)
        mask.save(img_name + "_mask.png")
    return

def load_trained_model(checkpoint):
    """
    loads a trained model
    """
    model = smp.Unet('resnet34', classes=1, aux_params=aux_params)
    try:
        model, executed_epochs = load_checkpoint(checkpoint, model)
    except Exception as e:
        print('No valid checkpoint has been found {}'. format(e))
    model.to(DEVICE)
    return model


def main():
    """
    Main function
    Runs in two different modes, training vs predicting
    """
    args = get_arguments()
    print("Choose folder for cell segmentation training data")
    rel_datapath = fd.askdirectory()
    print("Training data path: ", str(rel_datapath))
    print("Choose folder for cell segmentation output directory")
    output_dir = fd.askdirectory()
    print("Output path: ", str(output_dir))
    print("Choose file for checkpoint pkl")
    checkpoint = fd.askopenfilename()
    print("Checkpoint file: ", str(checkpoint))
    print("Choose folder for unlabeled data")
    unlabeled_dir = fd.askdirectory()
    print("Unlabeled data: ", str(unlabeled_dir))
    epochs     = args.epochs
    train_flag = args.train

    if train_flag:
        model, epochs, train_loss, val_loss = train_seg_model(rel_datapath, output_dir, checkpoint, epochs, args)
        draw_training_curves(train_loss, val_loss, epochs, output_dir)
    if args.validation:
        model = load_trained_model(checkpoint)
        run_inference_on_validation(rel_datapath,output_dir, model)
    if args.inference:
        model = load_trained_model(checkpoint)
        run_inference(unlabeled_dir, model)

    return





if __name__ == '__main__':
	main()
