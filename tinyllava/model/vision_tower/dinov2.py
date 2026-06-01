from transformers import Dinov2Model, AutoImageProcessor

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('facebook/dinov2-large')      
class DINOv2VisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = Dinov2Model(cfg)
        #self._image_processor = AutoImageProcessor.from_pretrained(cfg.model_name_or_path, size=336,crop_size={"height":336,"width":336})
        self._image_processor = AutoImageProcessor.from_pretrained(cfg.model_name_or_path, size=384,crop_size={"height":384,"width":384})
  
