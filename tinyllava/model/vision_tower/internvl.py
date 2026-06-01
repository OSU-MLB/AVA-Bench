from . import register_vision_tower
from .base import VisionTower
import torch.nn.functional as F
import torch

from transformers import AutoModel, CLIPImageProcessor

@register_vision_tower('InternViT-300M-448px-V2_5')      
class InternViTVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = AutoModel.from_pretrained(
                                cfg.model_name_or_path,
                                trust_remote_code=True)
        self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path,trust_remote_code=True)
   
    