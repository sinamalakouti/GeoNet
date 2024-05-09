import torchvision
from clip.clip import BICUBIC, _convert_image_to_rgb

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor


# class ResizeImage():
#     def __init__(self, size):
#         if isinstance(size, int):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#     def __call__(self, img):
#         th, tw = self.size
#         return img.resize((th, tw))


# class PlaceCrop(object):

#     def __init__(self, size, start_x, start_y):
#         if isinstance(size, int):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#         self.start_x = start_x
#         self.start_y = start_y

#     def __call__(self, img):
#         th, tw = self.size
#         return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

# def val_transform():
#     resize_size=256
#     crop_size=224
#     start_center = (resize_size - crop_size - 1) / 2
#     return torchvision.transforms.Compose([
#         ResizeImage(resize_size),
#         PlaceCrop(crop_size, start_center, start_center),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# def train_transform():
#     resize_size=256
#     crop_size=224
#     return torchvision.transforms.Compose([ResizeImage(resize_size),
#                   torchvision.transforms.RandomResizedCrop(crop_size),
#                   torchvision.transforms.RandomHorizontalFlip(),
#                   torchvision.transforms.ToTensor(),
#                   torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def train_transform():

    trans = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(256),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return trans

def val_transform():

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return trans

def test_transform():

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return trans

def clip_transform():
    return Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
def transform(split):

    trans = {
        "train" : clip_transform(),
        "val" : clip_transform(),
        "test" : clip_transform()
    }
    
    try:
        return trans[split]
    except:
        raise BaseException("Transforms for split {} not available, Choose [train/val/test]".format(split))
