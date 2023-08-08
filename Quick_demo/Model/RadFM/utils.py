from .blocks import ModifiedResNet,PMC_CLIP_cfg
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


    
def get_visual_encoder(model_str):
    """
    Args:
        str (_type_): str_to_model_path
    Return:
        vision_model, visual_dim, img_preprocessor
    """
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img_preprocessor = transforms.Compose([                        
                transforms.Resize((512,512), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
    if  'PMC-CLIP' in model_str:
        #vision_cfg = json.load(open(model_args.visual_model_config,'r'))['vision_cfg']
        vision_cfg = PMC_CLIP_cfg()
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        vision_model = ModifiedResNet(
            layers=vision_cfg.layers,
            heads=vision_heads,
            output_dim = 768,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width
        )
        vision_model = vision_load_pretrain(vision_model,model_str)
        vision_model = nn.Sequential(*list(vision_model.children())[:-2])
        visual_dim = 1024
    return vision_model,visual_dim,img_preprocessor

def vision_load_pretrain(resnet,model_path):
    checkpoint = torch.load(model_path, map_location='cpu') 
    state_dict = checkpoint['state_dict'] 
    state_dict = {k.replace('module.visual.',''): v for k, v in state_dict.items() if '.visual' in k}
    resnet.load_state_dict(state_dict)
    return resnet  
