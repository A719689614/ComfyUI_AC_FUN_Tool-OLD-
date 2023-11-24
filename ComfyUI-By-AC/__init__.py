from .AC_Math import *
from .AC_Controlnet import *
from .AC_Picture import *
from .AC_Sample import *
from .AC_Text import *

NODE_CLASS_MAPPINGS = {
    # 啊程的数学运算器
    "加法运算": Sum_AC,
    "减法运算": Sub_AC,
    "乘法运算": Mul_AC,
    "除法运算": Div_AC,
    "平方运算": Square_AC,
    "数学累加": Mathaddtion,
    "等差数列": Series_AC,
    "X变量值" : X_AC,
    "Y变量值" : Y_AC,
    "浮点数转整数":Translate_Float_AC,
    "整数转浮点数": Translate_INT_AC,
    "数字转STR文字":Translate_STR_AC,
    "文字显示面板":ShowText_AC,
    # 啊程的控制网格
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
    # 啊程的图片工具
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
    # 啊程的采样器
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
    # 啊程的文本工具
    "啊程提示词文本框": CLIPTextEncode_AC,
    "文本模型加载":CLIPLoader,
    "啊程记事本":Notebook,
    "嵌入式文本":Text_AC,
    "跳过CLIP文本层":CLIPSetLastLayer_AC,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Sum_AC": "加法运算",
    "Sub_AC": "减法运算",
    "Mult_AC": "乘法运算",
    "DIV_AC": "除法运算",
    "Square_AC": "平方运算",
    "X_AC" :"X变量值",
    "Y_AC": "Y变量值",
    "Translate_Float_AC": "浮点数转整数",
    "Translate_INT_AC": "整数转浮点数",
    "Translate_STR_AC": "数字转STR文字",
    "ShowText_AC": "文字显示面板",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]