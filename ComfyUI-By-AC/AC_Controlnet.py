import torch
import os
import sys

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import folder_paths


def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)

MAX_RESOLUTION=8192

# 控制网格模型加载
class ControlNetLoader_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
                             "tips": ("STRING", {
                             "multiline": False,
                             "default": "控制网格模型(controlnet)加载"})}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet_ac"

    CATEGORY = "啊程控制网格"

    def load_controlnet_ac(self, control_net_name, tips=None):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = comfy.sd.load_controlnet(controlnet_path)
        return (controlnet,)
# 一般控制器
class ControlNetSimple_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet_ac"

    CATEGORY = "啊程控制网格"

    def apply_controlnet_ac(self, conditioning, control_net, image, strength):
        if strength == 0:
            return (conditioning, )

        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = True
            c.append(n)
        return (c, )

# 网格加载
class UNETLoader_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("unet"), ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet_ac"

    CATEGORY = "啊程控制网格"

    def load_unet_ac(self, unet_name):
        unet_path = folder_paths.get_full_path("unet", unet_name)
        model = comfy.sd.load_unet(unet_path)
        return (model,)

# 高级网格控制
class ControlNetApplyAdvanced_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_point": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_point": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    
    FUNCTION = "apply_controlnet_ac"

    CATEGORY = "啊程控制网格"

    def apply_controlnet_ac(self, positive, negative, control_net, image, strength, start_point, end_point):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (1.0 - start_point, 1.0 - end_point))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])

# 区域控制
class ConditioningSetArea_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "width": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 1}),
                              "x_aera": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "y_aera": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "tips": ("STRING", {
                    "multiline": True,
                    "default": "啊程使用提醒!设置你绘图区域的X/Y坐标轴,用一张512X512图为例"
                    "它的原始坐标X/Y为(0,0),如果扩展面积为512X512,则X/Y(512,0)"
                             })},}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append_ac"

    CATEGORY = "啊程控制网格"

    def append_ac(self, conditioning, width, height, x_aera, y_aera, strength,tips=None):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['area'] = (height // 8, width // 8, y_aera // 8, x_aera // 8)
            n[1]['strength'] = strength
            n[1]['set_area_to_bounds'] = False
            c.append(n)
        return (c, )

# 合并条件
class ConditioningCombine_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_1": ("CONDITIONING", ), "conditioning_2": ("CONDITIONING", ),
                             "conditioning_3": ("CONDITIONING",),
                             "conditioning_4": ("CONDITIONING",),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine_ac"

    CATEGORY = "啊程控制网格"

    def combine(self, conditioning_1, conditioning_2,conditioning_3,conditioning_4):
        return (conditioning_1 + conditioning_2 + conditioning_3 + conditioning_4  )

# 求条件的平均值
class ConditioningAverage_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_to": ("CONDITIONING", ), "conditioning_from": ("CONDITIONING", ),
                              "conditioning_to_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "tips": ("STRING", {"multiline": False,
                              "default":"条件1到条件2的平均值"})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "addWeighted_ac"

    CATEGORY = "啊程控制网格"

    def addWeighted_ac(self, conditioning_to, conditioning_from, conditioning_to_strength,tips=None):
        out = []

        if len(conditioning_from) > 1:
            print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            t_to = conditioning_to[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)
        return (out, )

# 合并多组条件
class ConditioningConcat_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning_to": ("CONDITIONING",),
            "conditioning_from": ("CONDITIONING",),
            "tips": ("STRING", {"multiline": False,
                              "default":"合并多个条件"})
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "concat_ac"

    CATEGORY = "啊程控制网格"

    def concat_ac(self, conditioning_to, conditioning_from,tips=None):
        out = []

        if len(conditioning_from) > 1:
            print("Warning: ConditioningConcat conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            tw = torch.cat((t1, cond_from),1)
            n = [tw, conditioning_to[i][1].copy()]
            out.append(n)

        return (out, )

# 条件遮罩
class ConditioningSetMask_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "mask": ("MASK", ),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "set_cond_area": (["default", "mask bounds"],),
                              "tips": ("STRING", {"multiline": False,
                              "default":"设置条件的遮罩区域和强度"}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append_ac"

    CATEGORY = "啊程控制网格"

    def append_ac(self, conditioning, mask, set_cond_area, strength,tips=None):
        c = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            _, h, w = mask.shape
            n[1]['mask'] = mask
            n[1]['set_area_to_bounds'] = set_area_to_bounds
            n[1]['mask_strength'] = strength
            c.append(n)
        return (c, )

# 高级条件遮罩
class ConditioningSetTimestepRange_AC:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "tips": ("STRING", {"multiline": False,
                              "default":"设置条件的遮罩开始与结束!"}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_range_ac"

    CATEGORY = "啊程控制网格"

    def set_range_ac(self, conditioning, start, end,tips=None):
        c = []
        for t in conditioning:
            d = t[1].copy()
            d['start_percent'] = 1.0 - start
            d['end_percent'] = 1.0 - end
            n = [t[0], d]
            c.append(n)
        return (c, )

# 节点说明
NODE_CLASS_MAPPINGS = {
    "控制器加载": ControlNetLoader_AC,
    "高级控制网格": ControlNetApplyAdvanced_AC,
    "区域网格绘图控制": ConditioningSetArea_AC,
    "条件合并": ConditioningCombine_AC,
    "条件平均值":ConditioningAverage_AC,
    "合并多组条件":ConditioningConcat_AC,
    "条件遮罩":ConditioningSetMask_AC,
    "高级条件遮罩":ConditioningSetTimestepRange_AC,
    "一般控制网格": ControlNetSimple_AC,
    "网格加载": UNETLoader_AC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
        "ControlNetLoader_AC": "控制器加载",
        "ControlNetApplyAdvanced_AC": "高级控制网格",
        "ConditioningSetArea_AC": "区域网格绘图控制",
        "ConditioningCombine_AC": "条件合并",
        "ConditioningAverage_AC": "条件平均值",
        "ConditioningConcat_AC": "合并多组条件",
        "ConditioningSetMask_AC": "条件遮罩",
        "ConditioningSetTimestepRange_AC": "高级条件遮罩",
        "ControlNetSimple_AC": "一般控制网格",
        "UNETLoader_AC": "网格加载",
}



# 代码测试
if __name__ == "__main__":
    pass
