from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('google/siglip2-so400m-patch14-384')      
class SIGLIP2VisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = SiglipVisionModel(cfg)
        self._image_processor = SiglipImageProcessor.from_pretrained(cfg.model_name_or_path)
        
        
