import os
os.environ['HF_HOME'] = '/work/hdd/bcgd/duyan2/.cache'
import cv2
import json
import torch
import random
import numpy as np

from PIL import Image
from torchvision import transforms
from insightface.app import FaceAnalysis
from diffusers.utils import load_image

from utils.util import draw_kps

# Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, size=512,
                 dtype=torch.float16):
        '''
        for InstantID, never drop image embed
        '''
        super().__init__()

        self.size = size
        self.dtype = dtype

        self.data = json.load(open(json_file)) # list of dict

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def resize_img(self, input_image, max_side=1280, min_side=1024, size=None, 
                pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

        w, h = input_image.size
        if size is not None:
            w_resize_new, h_resize_new = size
        else:
            ratio = min_side / min(h, w)
            w, h = round(ratio*w), round(ratio*h)
            ratio = max_side / max(h, w)
            input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
            w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
            h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
        input_image = input_image.resize([w_resize_new, h_resize_new], mode)

        if pad_to_max_side:
            res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
            offset_x = (max_side - w_resize_new) // 2
            offset_y = (max_side - h_resize_new) // 2
            res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
            input_image = Image.fromarray(res)
        return input_image

    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["text"]
        image_file = item["image_file"]
        
        face_image = load_image(image_file)
        # face_image = self.resize_img(face_image)

        # extract face img with drawn landmarks
        face_kps = draw_kps(face_image, item["kps"])
        
        
        return {
            # "image": self.transform(face_image),
            "image": face_image,
            "image_control": [face_kps],
            "text": text
        }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    image_controls = [example["image_control"][0] for example in data]
    texts = [example["text"] for example in data]

    return {
        "images": images,
        "image_controls": image_controls,
        "texts": texts
    }

if __name__ == '__main__':
    dts = CustomDataset(json_file='../data/CelebA_sm/data_sm.json')