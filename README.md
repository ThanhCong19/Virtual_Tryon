<div align="center">

<h1>FPT-VTON: FPT Virtual Try-On with Diffusion Model</h1>

<a href='https://fptvton.vercel.app/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://huggingface.co/spaces/basso4/FPT-VTON'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow'></a>
<a href='https://huggingface.co/basso4/FPT_DM/blob/main/fptdm.ckpt'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

</div>



## Abstract

> This project focuses on image-based virtual try-on, which generates an image of a person wearing a selected garment by using two input images: one of the person and one of the garment. Previous research has adapted exemplar-based inpainting diffusion models to enhance the natural appearance of virtual try-on images compared to other approaches (such as GAN-based methods). However, these methods often fail to maintain the original garment’s identity. To address this limitation, we propose a new diffusion model, called FPT Virtual Try-on with Diffusion Model (FPT-VTON), designed to improve garment fidelity and produce realistic virtual try-on images. To enhance controllability, a base diffusion-based virtual try-on network was created, integrating ControlNet to add additional control conditions and enhance feature extraction for garment images. FPT-VTON is an upgrade of Garment-Conditioned Diffusion Model (GC-DM), as we reuse the inherent diffusion model architecture of CAT-DM paper and apply more Garment features are extracted using a CLIP transformer encoder for images, then combined with the garment’s detailed caption (e.g., “[V]: short sleeve round neck t-shirt”) processed through a CLIP transformer encoder for text, put them into cross-attention mechanism to combine together. The garment caption and is provided from in json format.

## Hardware Requirement

Our experiments were conducted on two NVIDIA GeForce RTX 4090 graphics cards, with a single RTX 4090 having 24GB of video memory. Please note that our model cannot be trained on graphics cards with less video memory than the RTX 4090.

## Environment Requirement

1.   Clone the repository

```bash
git clone https://github.com/ThanhCong19/Virtual_Tryon
```

2.   A suitable `conda` environment named `Virtual_Tryon` can be created and activated with:

```bash
cd Virtual_Tryon
conda env create -f environment.yaml
conda activate FPT-VTON
```

-   If you want to change the name of the environment you created, you need to modify the `name` in both `environment.yaml` and `setup.py`.
-   You need to make sure that `conda` is installed on your computer.
-   If there is a network error, try updating the environment using `conda env update -f environment.yaml`.

3.   Installing xFormers：

```bash
# [linux only] cuda 11.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
# [linux only] cuda 12.1 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
# [linux & win] cuda 12.4 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124
# [linux only] (EXPERIMENTAL) rocm 6.1 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/rocm6.1
```

4.   open `src/taming-transformers/taming/data/utils.py`, delete `from torch._six import string_classes`, and change `elif isinstance(elem, string_classes):` to `elif isinstance(elem, str):`

## Dataset Preparing

### VITON-HD

1.  Download the [VITON-HD](https://drive.google.com/file/d/1r8Ds39KQzGhesgdz8nrf8u_weOHRVDTe/view?usp=sharing) dataset
2.  Unzip it
3.  Generate the mask images

```bash
# Generate the train dataset mask images
python tools/mask_vitonhd.py datasets/vitonhd/train datasets/vitonhd/train/mask
# Generate the test dataset mask images
python tools/mask_vitonhd.py datasets/vitonhd/test datasets/vitonhd/test/mask
```

### Details
`datasets` folder should be as follows:

```
datasets
├── vitonhd
│   ├── test
│   │   ├── agnostic-mask
│   │   ├── mask
│   │   ├── cloth
│   │   ├── image
│   │   ├── image-densepose
│   │   ├── ...
│   ├── test_pairs.txt
│   ├── train
│   │   ├── agnostic-mask
│   │   ├── mask
│   │   ├── cloth
│   │   ├── image
│   │   ├── image-densepose
│   │   ├── ...
│   └── train_pairs.txt

```
PS: When we conducted the experiment, VITON-HD did not release the `agnostic-mask`. We used our own implemented `mask`, so if you are using VITON-HD's `agnostic-mask`, the generated results may vary.


## Required Model

1. Download the [Paint-by-Example](https://drive.google.com/file/d/1fHrsXmdcsLKEtp3Po8Q-6ZsweJWGAhHb/view?usp=sharing) model
2. Create a folder `checkpoints`
3. Put the Paint-by-Example model into this folder and rename it to `pbe.ckpt`
4. Make the ControlNet model:

- VITON-HD:
```bash
python tools/add_control.py checkpoints/pbe.ckpt checkpoints/pbe_dim6.ckpt configs/train_vitonhd.yaml
```

5.   `checkpoints` folder should be as follows:

```
checkpoints
├── pbe.ckpt
└── pbe_dim6.ckpt
```


## Training

### VITON-HD

```bash
bash scripts/train_vitonhd.sh
```


## Testing

### VITON-HD

1. Download the [checkpoint](https://huggingface.co/basso4/FPT_DM/resolve/main/fptdm.ckpt) for VITON-HD dataset and put it into `checkpoints` folder.

2. Directly generate the try-on results:

```bash
bash scripts/test_vitonhd.sh
```

3. Poisson Blending

```python
python tools/poisson_vitonhd.py
```


## Start a local gradio demo <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>

Download checkpoints for human parsing [here](https://huggingface.co/spaces/yisol/IDM-VTON-local/tree/main/ckpt).

Place the checkpoints under the ckpt folder.
```
ckpt
|-- densepose
    |-- model_final_162be9.pkl
|-- humanparsing
    |-- parsing_atr.onnx
    |-- parsing_lip.onnx

|-- openpose
    |-- ckpts
        |-- body_pose_model.pth
    
```


Run the following command:

```python
python gradio_demo/app.py
```


## Evaluation


1. Fréchet Inception Distance (FID)
- FID: https://github.com/mseitzer/pytorch-fid
- Measures distribution similarity between real and generated images.
- Strengths: Captures mean and covariance differences, widely used in GAN evaluation.
- Weaknesses: Computationally expensive, sensitive to resolution.

2. Kernel Inception Distance (KID)
- KID: https://github.com/toshas/torch-fidelity
- Similar to FID but uses unbiased kernel-based estimation.
- Strengths: Robust on small datasets.
- Weaknesses: Sensitive to kernel hyperparameters.

3. Structural Similarity Index Measure (SSIM)
- SSIM: https://github.com/richzhang/PerceptualSimilarity
- Assesses image quality based on luminance, contrast, and structure.
- Strengths: Efficient and interpretable.
- Weaknesses: Poor correlation with human perception for complex images.

4. Learned Perceptual Image Patch Similarity (LPIPS)
- LPIPS: https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html
- Measures perceptual similarity using deep feature maps.
- Strengths: Aligns well with human judgment.
- Weaknesses: Computationally expensive, depends on pretrained models.



## Acknowledgements


Thanks [ZeroGPU](https://huggingface.co/zero-gpu-explorers) for providing free GPU.

Thanks [CAT-DM](https://github.com/zengjianhao/CAT-DM) for base codes.

Thanks [IDM-VTON](https://github.com/yisol/IDM-VTON) for caption text.

Thanks [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) for human segmentation.

Thanks [Densepose](https://github.com/facebookresearch/DensePose) for human densepose.

