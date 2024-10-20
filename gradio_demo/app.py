import os
import sys
sys.path.append('./')
import numpy as np
import argparse

import torch
import torchvision
import pytorch_lightning
from torch import autocast
from torchvision import transforms
from pytorch_lightning import seed_everything


from einops import rearrange 
from functools import partial
from omegaconf import OmegaConf

from PIL import Image
from typing import List
import matplotlib.pyplot as plt

import gradio as gr
import apply_net

from torchvision.transforms.functional import to_pil_image
# from tools.mask_vitonhd import get_img_agnostic

from utils_mask import get_mask_location
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from ldm.util import instantiate_from_config, get_obj_from_str
from ldm.models.diffusion.ddim import DDIMSampler
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation


def un_norm(x):
    return (x+1.0)/2.0


def un_norm_clip(x):
    x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
    x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
    x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
    return x


class DataModuleFromConfig(pytorch_lightning.LightningDataModule):
    def __init__(self, 
                 batch_size, 
                 test=None, 
                 wrap=False, 
                 shuffle=False,
                 shuffle_test_loader=False,
                 use_worker_init_fn=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.wrap = wrap
        self.datasets = instantiate_from_config(test)
        self.dataloader = torch.utils.data.Dataloader(self.datasets,
                                                      batch_size=self.batch_size,
                                                      num_workers=self.num_workers,
                                                      shuffle=shuffle,
                                                      use_worker_init_fn=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for demo model")
    parser.add_argument("-b", "--base", type=str, default=r"configs/test_vitonhd.yaml")
    parser.add_argument("-c", "--ckpt", type=str, default=r"ckpt/hitonhd.ckpt")
    parser.add_argument("-s", "--seed", type=str, default=42)
    parser.add_argument("-d", "--ddim", type=str, default=64)
    opt = parser.parse_args()

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.base}")
    # data = instantiate_from_config(config.data)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model = instantiate_from_config(config.model)
    # model.load_state_dict(torch.load(opt.ckpt, map_location="cpu")["state_dict"], strict=False)
    # model.cuda()
    # model.eval()
    # model = model.to(device)
    # sampler = DDIMSampler(model)

    precision_scope = autocast


    def start_tryon(human_img,garm_img):
        #load human image
        human_img = human_img.convert("RGB").resize((768,1024))

        #mask
        tensor_transfrom = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
        )

        parsing_model = Parsing(0)
        openose_model = OpenPose(0)
        openose_model.preprocessor.body_estimation.model.to(device)

        keypoints = openose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
        mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray+1.0)/2.0)
        # mask_gray.save(r'D:\Capstone_Project\cat_dm\gradio_demo\output\maskgray_output.png')

        #densepose    
        human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        args = apply_net.create_argument_parser().parse_args(('show', 
                                                            './configs/configs_densepose/densepose_rcnn_R_50_FPN_s1x.yaml', 
                                                            './ckpt/densepose/model_final_162be9.pkl', 
                                                            'dp_segm', '-v', 
                                                            '--opts', 
                                                            'MODEL.DEVICE', 
                                                            'cuda'))
        # verbosity = getattr(args, "verbosity", None)
        pose_img = args.func(args,human_img_arg)    
        pose_img = pose_img[:,:,::-1]    
        pose_img = Image.fromarray(pose_img).resize((768,1024))

        #preprocessing image
        human_img = human_img.convert("RGB").resize((512, 512))
        human_img = torchvision.transforms.ToTensor()(human_img)

        garm_img = garm_img.convert("RGB").resize((224, 224))
        garm_img = torchvision.transforms.ToTensor()(garm_img)

        mask = mask.convert("L").resize((512,512))
        mask = torchvision.transforms.ToTensor()(mask)
        mask = 1-mask

        pose_img = pose_img.convert("RGB").resize((512, 512))
        pose_img = torchvision.transforms.ToTensor()(pose_img)

        #Normalize
        human_img = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(human_img)
        garm_img = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                    (0.26862954, 0.26130258, 0.27577711))(garm_img)
        pose_img = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(pose_img)

        #create inpaint & hint
        inpaint = human_img * mask
        hint = torchvision.transforms.Resize((512, 512))(garm_img)
        hint = torch.cat((hint, pose_img), dim=0)

        # {"human_img": human_img,     # [3, 512, 512]
        # "inpaint_image": inpaint,   # [3, 512, 512]
        # "inpaint_mask": mask,       # [1, 512, 512]
        # "garm_img": garm_img,       # [3, 224, 224]
        # "hint": hint,               # [6, 512, 512]
        # }


        with torch.no_grad():
            with precision_scope("cuda"):
                #loading data
                inpaint = inpaint.unsqueeze(0).to(torch.float16).to(device)
                reference = garm_img.unsqueeze(0).to(torch.float16).to(device)
                mask = mask.unsqueeze(0).to(torch.float16).to(device)
                hint = hint.unsqueeze(0).to(torch.float16).to(device)
                truth = human_img.unsqueeze(0).to(torch.float16).to(device)

                #data preprocessing
                encoder_posterior_inpaint = model.first_stage_model.encode(inpaint)
                z_inpaint = model.scale_factor * (encoder_posterior_inpaint.sample()).detach()
                mask_resize = torchvision.transforms.Resize([z_inpaint.shape[-2],z_inpaint.shape[-1]])(mask)
                test_model_kwargs = {}
                test_model_kwargs['inpaint_image'] = z_inpaint
                test_model_kwargs['inpaint_mask'] = mask_resize
                shape = (model.channels, model.image_size, model.image_size)

                #predict
                samples, _ = sampler.sample(S=opt.ddim,
                                                 batch_size=1,
                                                 shape=shape,
                                                 pose=hint,
                                                 conditioning=reference,
                                                 verbose=False,
                                                 eta=0,
                                                 test_model_kwargs=test_model_kwargs)
                samples = 1. / model.scale_factor * samples
                x_samples = model.first_stage_model.decode(samples[:,:4,:,:])

                x_samples_ddim = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                x_checked_image=x_samples_ddim
                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                # Xử lý và trả về img và img_C
                mask = mask.cpu().permute(0, 2, 3, 1).numpy()
                mask = torch.from_numpy(mask).permute(0, 3, 1, 2)
                truth = torch.clamp((truth + 1.0) / 2.0, min=0.0, max=1.0)
                truth = truth.cpu().permute(0, 2, 3, 1).numpy()
                truth = torch.from_numpy(truth).permute(0, 3, 1, 2)

                x_checked_image_torch_C = x_checked_image_torch * (1 - mask) + truth.cpu() * mask
                x_checked_image_torch = torch.nn.functional.interpolate(x_checked_image_torch.float(), size=[512, 384])
                x_checked_image_torch_C = torch.nn.functional.interpolate(x_checked_image_torch_C.float(), size=[512, 384])

                # Tạo và trả về hình ảnh img và img_C
                img = x_checked_image_torch[0].cpu().numpy().transpose(1, 2, 0)  # Chuyển về định dạng HWC
                img_C = x_checked_image_torch_C[0].cpu().numpy().transpose(1, 2, 0)  # Chuyển về định dạng HWC

                return img, img_C, mask_gray


example_path = os.path.join(os.path.dirname(__file__), 'example')

garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]



image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## CAT-DM 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image")
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(sources='upload', type="pil", label='Human Picture or use Examples below', interactive=True)

            example = gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                examples=human_list_path
            )
        
        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")

            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path
            )
        
        with gr.Column():
            image_out = gr.Image(label="Output", elem_id="output-img",show_download_button=False)
            try_button = gr.Button(value="Try-on")

        with gr.Column():
            image_out_c = gr.Image(label="Output", elem_id="output-img",show_download_button=False)

        with gr.Column():
            masked_img = gr.Image(label="Masked image output", elem_id="masked_img", show_download_button=False)


    try_button.click(fn=start_tryon, inputs=[imgs,garm_img], outputs=[image_out,image_out_c,masked_img], api_name='tryon')



image_blocks.launch()
