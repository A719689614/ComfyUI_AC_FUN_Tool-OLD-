import torch

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


import comfy.diffusers_load
import comfy.samplers as AC
import comfy.sample  as ac
import comfy.sd
import comfy.utils

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths as fp
import latent_preview


def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)

MAX_RESOLUTION=8192

# 采样器函数
def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = latent_preview.get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )

# 采样器
class KSamplerAC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "add_noise": (["enable", "disable"], ),
                     "denoise": ("FLOAT",{"default": 1.0,"min": 0.0, "max": 1.0,"step": 0.1}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (AC.KSampler.SAMPLERS, ),
                    "scheduler": (AC.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 30, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),

                     }
                }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "SampleAC"

    CATEGORY = "啊程采样器"

    def SampleAC(self, model, add_noise, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise,
                                  start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

# 啊程模型加载器
class CheckpointLoaderAC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ckpt_name": (fp.get_filename_list("checkpoints"), )}}
    RETURN_TYPES = ("MODEL","CLIP", "VAE")
    FUNCTION = "load_checkpointAC"

    CATEGORY = "啊程模型和Lora加载"

    def load_checkpointAC(self, ckpt_name, output_vae=True, output_clip=True,):
        ckpt_path = fp.get_full_path("checkpoints", ckpt_name)
        out_01 = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=fp.get_folder_paths("embeddings"))
        
        return out_01


# 啊程编码器
class VAEEncode_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode_ac"

    CATEGORY = "啊程采样器"

    @staticmethod
    def vae_encode_crop_pixels_ac(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def encode_ac(self, vae, pixels):
        pixels = self.vae_encode_crop_pixels_ac(pixels)
        t = vae.encode(pixels[:,:,:,:3])
        return ({"samples":t}, )

# 啊程VAE加载
class VAELoader_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (fp.get_filename_list("vae"), )}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae_ac"

    CATEGORY = "啊程采样器"

    #TODO: scale factor?
    def load_vae_ac(self, vae_name):
        vae_path = fp.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(ckpt_path=vae_path)
        return (vae,)

# 啊程解码器
class VAEDecode_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode_ac"

    CATEGORY = "啊程采样器"

    def decode_ac(self, vae, samples):
        return (vae.decode(samples["samples"]), )
# 啊程重绘编码器
class VAEEncodeForInpaint_ac:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("VAE", ), "mask": ("MASK", ), "羽化值": ("INT", {"default": 5, "min": 0, "max": 128, "step": 1}),}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode_ac"

    CATEGORY = "啊程采样器"

    def encode_ac(self, vae, pixels, mask, 羽化值=5):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        #grow mask by a few pixels to keep things seamless in latent space
        if 羽化值 == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, 羽化值, 羽化值))
            padding = math.ceil((羽化值 - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        t = vae.encode(pixels)

        return ({"samples":t, "noise_mask": (mask_erosion[:,:,:x,:y].round())}, )

# 加载默认文件
class DiffusersLoader_AC:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in fp.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {"required": {"model_path": (paths,), }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint_ac"

    CATEGORY = "啊程模型和Lora加载"

    def load_checkpoint_ac(self, model_path, output_vae=True, output_clip=True):
        for search_path in fp.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, model_path)
                if os.path.exists(path):
                    model_path = path
                    break

        return comfy.diffusers_load.load_diffusers(model_path, fp16=comfy.model_management.should_use_fp16(), output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))

# Lora模型加载
class LoraLoader_AC:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (fp.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "tips": ("STRING", {"multiline": False,
                              "default":"加载Lora风格文件,下载地址:liblib.ai"}),
                              
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora_ac"

    CATEGORY = "啊程模型和Lora加载"

    def load_lora_ac(self, model, clip, lora_name, strength_model, strength_clip,tips=None):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = fp.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

# CLIPVISION 加载
class CLIPVisionLoader_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (fp.get_filename_list("clip_vision"), ),
                             }}
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip_ac"

    CATEGORY = "啊程模型和Lora加载"

    def load_clip_ac(self, clip_name):
        clip_path = fp.get_full_path("clip_vision", clip_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        return (clip_vision,)

# CLIPVISION 编码
class CLIPVisionEncode_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "image": ("IMAGE",)
                             }}
    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    FUNCTION = "encode_ac"

    CATEGORY = "啊程采样器"

    def encode_ac(self, clip_vision, image):
        output = clip_vision.encode_image(image)
        return (output,)


# 加载潜空间
class LoadLatent_AC:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = fp.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".latent")]
        return {"required": {"latent": [sorted(files), ]},
                "tips": ("STRING", {"multiline": False,
                              "default":"加载一个潜空间"}), }

    CATEGORY = "啊程模型和Lora加载"

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "load_ac"

    def load_ac(self, latent,tips=None):
        latent_path = fp.get_annotated_filepath(latent)
        latent = safetensors.torch.load_file(latent_path, device="cpu")
        multiplier = 1.0
        if "latent_format_version_0" not in latent:
            multiplier = 1.0 / 0.18215
        samples = {"samples": latent["latent_tensor"].float() * multiplier}
        return (samples, )

# 潜变量反转
class LatentFlip_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "flip_method": (["x-axis: vertically", "y-axis: horizontally"],),
                              "tips": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "反转你的潜空间的X轴和Y轴,你可以理解为水平出图或者垂直出图!"}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "flip_ac"

    CATEGORY = "啊程采样器"

    def flip_ac(self, samples, flip_method,tips=None):
        s = samples.copy()
        if flip_method.startswith("x"):
            s["samples"] = torch.flip(samples["samples"], dims=[2])
        elif flip_method.startswith("y"):
            s["samples"] = torch.flip(samples["samples"], dims=[3])

        return (s,)

# 类别模型加载
class StyleModelLoader_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "style_model_name": (fp.get_filename_list("style_models"), )}}

    RETURN_TYPES = ("STYLE_MODEL",)
    FUNCTION = "load_style_model_ac"

    CATEGORY = "啊程模型和Lora加载"

    def load_style_model_ac(self, style_model_name):
        style_model_path = fp.get_full_path("style_models", style_model_name)
        style_model = comfy.sd.load_style_model(style_model_path)
        return (style_model,)

# 风格模型应用
class StyleModelApply_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel_ac"

    CATEGORY = "啊程模型和Lora加载"

    def apply_stylemodel_ac(self, clip_vision_output, style_model, conditioning):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, )    
# 潜空间放大
class LatentUpscale_AC:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",), "upscale_method": (s.upscale_methods,),
                              "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale_ac"

    CATEGORY = "啊程模型和Lora加载"

    def upscale_ac(self, samples, upscale_method, width, height, crop):
        s = samples.copy()
        s["samples"] = comfy.utils.common_upscale(samples["samples"], width // 8, height // 8, upscale_method, crop)
        return (s,)  
    
# 潜空间放大伴随模型
class LatentUpscaleBy_AC:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",), "upscale_method": (s.upscale_methods,),
                              "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale_ac"

    CATEGORY = "啊程模型和Lora加载"

    def upscale_ac(self, samples, upscale_method, scale_by):
        s = samples.copy()
        width = round(samples["samples"].shape[3] * scale_by)
        height = round(samples["samples"].shape[2] * scale_by)
        s["samples"] = comfy.utils.common_upscale(samples["samples"], width, height, upscale_method, "disabled")
        return (s,)
    
# 潜空间选择
class LatentRotate_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "rotation": (["none", "90 degrees", "180 degrees", "270 degrees"],),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "rotate_ac"

    CATEGORY = "啊程模型和Lora加载"

    def rotate(self, samples, rotation):
        s = samples.copy()
        rotate_by = 0
        if rotation.startswith("90"):
            rotate_by = 1
        elif rotation.startswith("180"):
            rotate_by = 2
        elif rotation.startswith("270"):
            rotate_by = 3

        s["samples"] = torch.rot90(samples["samples"], k=rotate_by, dims=[3, 2])
        return (s,)
# 潜空间反转
class LatentFlip_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "flip_method": (["x-axis: vertically", "y-axis: horizontally"],),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "flip_ac"

    CATEGORY = "啊程模型和Lora加载"

    def flip_ac(self, samples, flip_method):
        s = samples.copy()
        if flip_method.startswith("x"):
            s["samples"] = torch.flip(samples["samples"], dims=[2])
        elif flip_method.startswith("y"):
            s["samples"] = torch.flip(samples["samples"], dims=[3])

        return (s,)

# 潜空间合并
class LatentComposite_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples_to": ("LATENT",),
                              "samples_from": ("LATENT",),
                              "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "feather": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "tips": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "一个实验的潜空间复合器"}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite_ac"

    CATEGORY = "啊程采样器"

    def composite(self, samples_to, samples_from, x, y, composite_method="normal", feather=0,tips=None):
        x =  x // 8
        y = y // 8
        feather = feather // 8
        samples_out = samples_to.copy()
        s = samples_to["samples"].clone()
        samples_to = samples_to["samples"]
        samples_from = samples_from["samples"]
        if feather == 0:
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
        else:
            samples_from = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
            mask = torch.ones_like(samples_from)
            for t in range(feather):
                if y != 0:
                    mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))

                if y + samples_from.shape[2] < samples_to.shape[2]:
                    mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                if x != 0:
                    mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                if x + samples_from.shape[3] < samples_to.shape[3]:
                    mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
            rev_mask = torch.ones_like(mask) - mask
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x] * mask + s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] * rev_mask
        samples_out["samples"] = s
        return (samples_out,)

# 潜空间混合
class LatentBlend_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples1": ("LATENT",),
            "samples2": ("LATENT",),
            "blend_factor": ("FLOAT", {
                "default": 0.5,
                "min": 0,
                "max": 1,
                "step": 0.01
            }),
            "tips": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "一个实验的潜空间混合器"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "blend_ac"

    CATEGORY = "啊程采样器"

    def blend_ac(self, samples1, samples2, blend_factor:float, blend_mode: str="normal",tips=None):

        samples_out = samples1.copy()
        samples1 = samples1["samples"]
        samples2 = samples2["samples"]

        if samples1.shape != samples2.shape:
            samples2.permute(0, 3, 1, 2)
            samples2 = comfy.utils.common_upscale(samples2, samples1.shape[3], samples1.shape[2], 'bicubic', crop='center')
            samples2.permute(0, 2, 3, 1)

        samples_blended = self.blend_mode(samples1, samples2, blend_mode)
        samples_blended = samples1 * blend_factor + samples_blended * (1 - blend_factor)
        samples_out["samples"] = samples_blended
        return (samples_out,)

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")
# 节点命名
NODE_CLASS_MAPPINGS = {
    "啊程采样器": KSamplerAC,
    "啊程加载器": CheckpointLoaderAC,
    "啊程编码器": VAEEncode_AC,
    "啊程的VAE加载": VAELoader_AC,
    "啊程解码器":VAEDecode_AC,
    "啊程重绘编码器": VAEEncodeForInpaint_ac,
    "加载默认模型":DiffusersLoader_AC,
    "Lora加载器":LoraLoader_AC,
    "潜空间加载": LoadLatent_AC,
    "潜变量反转": LatentFlip_AC,
    "风格模型加载": StyleModelLoader_AC,
    "风格模型应用": StyleModelApply_AC,
    "潜空间放大": LatentUpscale_AC,
    "潜空间放大(模型)": LatentUpscaleBy_AC,
    "潜空间旋转": LatentRotate_AC,
    "潜空间反转": LatentFlip_AC,
    "标签模型加载": CLIPVisionLoader_AC,
    "标签模型解码": CLIPVisionEncode_AC,
    "潜空间合并": LatentComposite_AC,
    "潜空间混合": LatentBlend_AC,
    }
# 显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
        "KSampelrAC": "啊程采样器",
        "CheckpointLoaderAC": "啊程加载器",
        "VAEEncode_AC":"啊程编码器",
        "VAELoader_AC": "啊程的VAE加载",
        "VAEDecode_AC": "啊程解码器",
        "VAEEncodeForInpaint_ac": "啊程重绘编码器",
        "DiffusersLoader_AC": "加载默认模型",
        "LoraLoader_AC": "Lora加载器",
        "LoadLatent_AC": "潜空间加载",
        "LatentFlip_AC": "潜变量反转",
        "StyleModelLoader_AC": "风格模型加载",
        "StyleModelApply_AC": "风格模型应用",
        "LatentUpscale_AC": "潜空间放大",
        "LatentUpscaleBy_AC": "潜空间放大(模型)",
        "LatentRotate_AC": "潜空间旋转",
        "LatentFlip_AC": "潜空间反转",
        "CLIPVisionLoader_AC":"标签模型加载",
        "CLIPVisionEncode_AC": "标签空间解码",
        "LatentComposite_AC": "潜空间合并",
        "LatentBlend_AC": "潜空间混合",

}