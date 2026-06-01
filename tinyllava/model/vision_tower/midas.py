from transformers import DPTImageProcessor, DPTForDepthEstimation

from . import register_vision_tower
from .base import VisionTower

@register_vision_tower('midas')      
class MidasVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg) 
        self._vision_tower = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas",low_cpu_mem_usage=False)
        self._image_processor = DPTImageProcessor.from_pretrained(cfg.model_name_or_path)
