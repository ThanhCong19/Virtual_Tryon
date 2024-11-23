import os
import sys
import cv2
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for demo model")
    parser.add_argument("-b", "--base", type=str, default=r"configs/test_vitonhd.yaml")
    parser.add_argument("-c", "--ckpt", type=str, default=r"ckpt/hitonhd.ckpt")
    parser.add_argument("-s", "--seed", type=str, default=42)
    parser.add_argument("-d", "--ddim", type=str, default=16)
    opt = parser.parse_args()

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.base}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model = instantiate_from_config(config.model)
    # model.load_state_dict(torch.load(opt.ckpt, map_location="cpu")["state_dict"], strict=False)
    # model.cuda()
    # model.eval()
    # model = model.to(device)
    # sampler = DDIMSampler(model)

    precision_scope = autocast


    def start_tryon(dict_human,garm_img):
        #load human image
        human_img = dict_human['background'].convert("RGB").resize((768,1024))

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
        mask_cv = mask
        mask = mask.resize((768, 1024))
        mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray+1.0)/2.0)
    
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
                
                x_checked_image_torch = torch.nn.functional.interpolate(x_checked_image_torch.float(), size=[512,384])

                #apply seamlessClone technique here
                #img_base
                dict_human = dict_human['background'].convert("RGB").resize((384, 512))
                dict_human = np.array(dict_human).astype(np.uint8)
                dict_human = cv2.cvtColor(dict_human, cv2.COLOR_RGB2BGR)

                #img_output
                img_cv = rearrange(x_checked_image_torch[0], 'c h w -> h w c').cpu().numpy()
                img_cv = (img_cv * 255).astype(np.uint8)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                
                #mask
                mask_cv = mask_cv.convert("L").resize((384,512))
                mask_cv = np.array(mask_cv).astype(np.uint8)
                mask_cv = 255-mask_cv
    
                img_C = cv2.seamlessClone(dict_human, img_cv, mask_cv, (192,256), cv2.NORMAL_CLONE)
                img_C = cv2.cvtColor(img_C, cv2.COLOR_BGR2RGB)
                
                return img_C, mask_gray


example_path = os.path.join(os.path.dirname(__file__), 'example')

garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict= {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

##default human


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## FPT_VTON 👕👔👚")
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
            image_out_c = gr.Image(label="Output", elem_id="output-img",show_download_button=True)
            try_button = gr.Button(value="Try-on")

        # with gr.Column():
        #     image_out_c = gr.Image(label="Output", elem_id="output-img",show_download_button=False)

        with gr.Column():
            masked_img = gr.Image(label="Masked image output", elem_id="masked_img", show_download_button=True)


    try_button.click(fn=start_tryon, inputs=[imgs,garm_img], outputs=[image_out_c,masked_img], api_name='tryon')



image_blocks.launch()
