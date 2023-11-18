import glob
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from skimage.color import gray2rgb, rgb2gray,rgba2rgb
from pathlib import Path

class CellData(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.masks = list(Path(datapath).glob("**/*_mask.png"))
        train_transforms = [ torchvision.transforms.ToTensor(),torchvision.transforms.Resize((256,256))]
        self.train_transforms = torchvision.transforms.Compose(train_transforms)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask_file = self.masks[idx]
        mask_img  = io.imread(mask_file)

        if len(mask_img.shape) >2:
            if mask_img.shape[2] == 4:
                mask_img = rgba2rgb(mask_img)
            mask_img = rgb2gray(mask_img)

        cell_path = str(mask_file.resolve()).strip("_mask.png") + ".png"
        cell_img_org  = io.imread(cell_path)
        cell_img  = gray2rgb(cell_img_org)
        cell_img = self.train_transforms(cell_img)
        mask_img = self.train_transforms(mask_img)

        return cell_img, mask_img, cell_img_org
