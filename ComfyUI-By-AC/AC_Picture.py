import torch

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
from torch import Tensor

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview


def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)

MAX_RESOLUTION=8192
# 啊程相册
class SaveImage_AC:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "啊程图集"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images_ac"

    OUTPUT_NODE = True

    CATEGORY = "啊程图像工具"    
    
    def save_images_ac(self, images, filename_prefix="啊程图集", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
    
# 图片预览    
class PreviewImage_Ac(SaveImage_AC):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    
#  虚拟空间分辨率设置
class EmptyLatentImage_AC:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 1}),
                              "amount": ("INT", {"default": 1, "min": 1, "max": 128}),
                              "print_to_screen": (["enabled","disabled"],),
                              "tips": ("STRING",
                              {"multiline": False, # 是否显示多行文本
                              "default": "请输入你预想的图片的分辨率大小,以64倍数为最佳效果"}),
                              
                            }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_ac"

    CATEGORY = "啊程图像工具"

    def generate_ac(self, width, height, amount=1,tips=None,print_to_screen= None):
        latent = torch.zeros([amount, 4, height // 8, width // 8])
        return ({"samples":latent},)


# 重置数量
class RepeatLatentBatch_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "amount": ("INT", {"default": 1, "min": 1, "max": 64}),
                              "tips": ("STRING",
                              {"multiline": False, # 是否显示多行文本
                              "default": "重新输入你想要的数量，以整数为例！"}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "repeat_ac"

    CATEGORY = "啊程图像工具"

    def repeat_ac(self, samples, amount,tips=None):
        s = samples.copy()
        s_in = samples["samples"]
        
        s["samples"] = s_in.repeat((amount, 1,1,1))
        if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
            masks = samples["noise_mask"]
            if masks.shape[0] < s_in.shape[0]:
                masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[:s_in.shape[0]]
            s["noise_mask"] = samples["noise_mask"].repeat((amount, 1,1,1))
        if "batch_index" in s:
            offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
            s["batch_index"] = s["batch_index"] + [x + (i * offset) for i in range(1, amount) for x in s["batch_index"]]
        return (s,)

# 加载图片
class LoadImage_AC:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})
                    
                     },
                }

    CATEGORY = "啊程图像工具"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_ac"
    def load_image_ac(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

# 获取图片尺寸
class Picturetool_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),},
                }
    RETURN_TYPES = ("INT","INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "picturetool_ac"

    CATEGORY = "啊程图像工具"
# TODO:
    def picturetool_ac(self,image: Tensor):
        # 获取当前工作目录
        ( height, width) = image.shape[1:3]
        return (width, height)
        






# 反转图片
class ImageInvert_AC:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                             "tips": ("STRING", {"multiline": False,
                              "default":"反转图片的色相,黑白色调对换!"}),},}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "invert_ac"

    CATEGORY = "啊程图像工具"

    def invert_ac(self, image,tips=None):
        invert = 1.0 - image
        return (invert,)

# 批量合并图片
class ImageBatch_AC:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image1": ("IMAGE",), "image2": ("IMAGE",),
                             "tips": ("STRING", {"multiline": False,
                              "default":"将两张图片合并到一起"}),}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_ac"

    CATEGORY = "啊程图像工具"

    def batch_ac(self, image1, image2,tips=None):
        if image1.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(image2.movedim(-1,1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1,-1)
        s = torch.cat((image1, image2), dim=0)
        return (s,)
# 合并多个图片
class ImageBatch_MUT:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image1": ("IMAGE",), "image2": ("IMAGE",),"image3":("IMAGE",),"image4":("IMAGE",),
                             "tips": ("STRING", {"multiline": False,
                              "default":"将四张图片合并到一起"}),}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_mut"

    CATEGORY = "啊程图像工具"

    def batch_mut(self, image1, image2,image3,image4,tips=None):
        if image1.shape[1:] and image3.shape[1:] and image4.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(image2.movedim(-1,1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1,-1)
        result = torch.cat((image1, image2,image3,image4), dim=0)
        return (result,)

# 加载图片遮罩
class LoadImageMask_AC:
    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), ),
                     "channel": (s._color_channels, ), }
                }

    CATEGORY = "啊程图像工具"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image_ac"
    def load_image_ac(self, image, channel):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        if i.getbands() != ("R", "G", "B", "A"):
            i = i.convert("RGBA")
        mask = None
        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            if c == 'A':
                mask = 1. - mask
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (mask,)

    @classmethod
    def IS_CHANGED_AC(s, image, channel):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS_AC(s, image, channel):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        if channel not in s._color_channels:
            return "Invalid color channel: {}".format(channel)

        return True
# 图片放大
class ImageScale_AC:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_ac"

    CATEGORY = "啊程图像工具"

    def upscale_ac(self, image, upscale_method, width, height, crop):
        samples = image.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        s = s.movedim(1,-1)
        return (s,)
# 图片放大(模型)
class ImageScaleB_AC:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_ac"

    CATEGORY = "啊程图像工具"

    def upscale_ac(self, image, upscale_method, scale_by):
        samples = image.movedim(-1,1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1,-1)
        return (s,)

# 功能区命名
NODE_CLASS_MAPPINGS = {
    "啊程图像预览": SaveImage_AC,
    "啊程分辨率设置": EmptyLatentImage_AC,
    "重置数量": RepeatLatentBatch_AC,
    "获取尺寸": Picturetool_AC,
    "反转图像":ImageInvert_AC,
    "双图合并": ImageBatch_AC,
    "多图像合并": ImageBatch_MUT,
    "加载图片遮罩": LoadImageMask_AC,
    "加载图片": LoadImage_AC,
    "图像放大": ImageScale_AC,
    "图像放大(模型)": ImageScaleB_AC,
    
}

# 节点显示命名
NODE_DISPLAY_NAME_MAPPINGS = {
        "SaveImage_AC":"啊程图像预览",
        "EmptyLatentImage_AC":"啊程分辨率设置",
        "RepeatLatentBatch_AC": "重置数量",
        "Picturetool_AC": "获取尺寸",
        "ImageInvert_AC": "反转图像",
        "ImageBatch_AC": "双图合并",
        "ImageBatch_MUT": "多图像合并",
        "LoadImageMask_AC": "加载图片遮罩",
        "LoadImage_AC": "加载图像",
        "ImageScale_AC": "图像放大",
        "ImageScaleB_AC": "图像放大(模型)"
}



