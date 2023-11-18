import torchvision
from skimage import io
from skimage.color import gray2rgb

def readimage(image):
    """
    Takes a raw 512 x 512 png and prepares it for the model
    Args:
        - image in array form
    Output:
        - cell_img - converted to tensor form
    """
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((256, 256)),
        ]
    )
    cell_img = gray2rgb(image)
    cell_img = train_transforms(cell_img)
    return cell_img
