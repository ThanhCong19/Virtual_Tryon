import os
import json
import random
import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Literal, Tuple,List


class OpenImageDataset(data.Dataset):
    def __init__(
        self, 
        state: Literal["train", "test"], 
        dataset_dir: str, 
        type: Literal["paired", "unpaired"] = "paired",
    ):
        self.state=state
        self.dataset_dir = dataset_dir
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)

        with open(
            os.path.join(dataset_dir, state, "vitonhd_" + state + "_tagged.json"), "r"
        ) as file1:
            data1 = json.load(file1)

        annotation_list = [
            # "colors",
            # "textures",
            "sleeveLength",
            "neckLine",
            "item",
        ]

        self.annotations_pair = {}
        for k, v in data1.items():
            for elem in v:
                annotation_str = ""
                for template in annotation_list:
                    for tag in elem["tag_info"]:  
                        if (
                            tag["tag_name"] == template
                            and tag["tag_category"] is not None
                        ):
                            annotation_str += tag["tag_category"]
                            annotation_str += " "
                self.annotations_pair[elem["file_name"]] = annotation_str          


        im_names = []
        c_names = []

        if state == "train":
            filename = os.path.join(dataset_dir, f"{state}_pairs.txt")
        else:
            filename = os.path.join(dataset_dir, f"{state}_pairs.txt")
        
        with open(filename, "r") as f:
            for line in f.readlines():
                if state == "train":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if type == "paired":
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()
                
                im_names.append(im_name)
                c_names.append(c_name)
        
        self.im_names = im_names
        self.c_names = c_names                    

    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        if c_name in self.annotations_pair:
            cloth_annotation = self.annotations_pair[c_name]
        else:
            cloth_annotation = "shirts"

        # 确定路径
        img_path = os.path.join(self.dataset_dir, self.state, "image", im_name)
        reference_path = os.path.join(self.dataset_dir, self.state, "cloth", c_name)
        mask_path = os.path.join(self.dataset_dir, self.state, "agnostic-mask", im_name[:-4]+"_mask.png")                              
        densepose_path = os.path.join(self.dataset_dir, self.state, "image-densepose", im_name)
        
        # 加载图像
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        img = torchvision.transforms.ToTensor()(img)
        reference = Image.open(reference_path).convert("RGB").resize((224, 224))
        reference = torchvision.transforms.ToTensor()(reference)
        mask = Image.open(mask_path).convert("L").resize((512, 512))
        mask = torchvision.transforms.ToTensor()(mask)
        mask = 1-mask
        densepose = Image.open(densepose_path).convert("RGB").resize((512, 512))
        densepose = torchvision.transforms.ToTensor()(densepose)


        #Data augmentation for training phase
        if self.state == "train":
            #Random horizontal flip
            if random.random() > 0.5:
                img = self.flip_transform(img)
                mask = self.flip_transform(mask)
                densepose = self.flip_transform(densepose)
                reference = self.flip_transform(reference)

            #Color jittering
            if random.random() > 0.5:
                color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5)
                fn_idx, b, c, s, h = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation, color_jitter.hue)

                img = TF.adjust_contrast(img, c)
                img = TF.adjust_brightness(img, b)
                img = TF.adjust_hue(img, h)
                img = TF.adjust_saturation(img, s)

                reference = TF.adjust_contrast(reference, c)
                reference = TF.adjust_brightness(reference, b)
                reference = TF.adjust_hue(reference, h)
                reference = TF.adjust_saturation(reference, s)

            #Scaling and shifting
            if random.random() > 0.5:
                scale_val = random.uniform(0.8, 1.2)
                img = transforms.functional.affine(
                    img, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                mask = transforms.functional.affine(
                    mask, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                densepose = transforms.functional.affine(
                    densepose, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )

            if random.random() > 0.5:
                shift_valx = random.uniform(-0.2, 0.2)
                shift_valy = random.uniform(-0.2, 0.2)
                img = transforms.functional.affine(
                    img, 
                    angle=0, 
                    translate=[shift_valx * img.shape[-1], shift_valy * img.shape[-2]], 
                    scale=1, 
                    shear=0
                )
                mask = transforms.functional.affine(
                    mask, 
                    angle=0, 
                    translate=[shift_valx * mask.shape[-1], shift_valy * mask.shape[-2]], 
                    scale=1, 
                    shear=0
                )
                densepose = transforms.functional.affine(
                    densepose, 
                    angle=0, 
                    translate=[
                        shift_valx * densepose.shape[-1], 
                        shift_valy * densepose.shape[-2]
                    ], 
                    scale=1, 
                    shear=0
                )



        # 正则化
        img = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        reference = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                    (0.26862954, 0.26130258, 0.27577711))(reference)
        densepose = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(densepose)

        # 生成 inpaint 和 hint
        inpaint = img * mask
        hint = torchvision.transforms.Resize((512, 512))(reference)
        hint = torch.cat((hint,densepose),dim = 0)

        return {"GT": img,                                          # [3, 512, 512]
                "inpaint_image": inpaint,                           # [3, 512, 512]
                "inpaint_mask": mask,                               # [1, 512, 512]
                "ref_imgs": reference,                              # [3, 224, 224]
                "hint": hint,                                       # [6, 512, 512]
                "caption_cloth": "a photo of " + cloth_annotation,
                # "caption": "model is wearing " + cloth_annotation,
                }



