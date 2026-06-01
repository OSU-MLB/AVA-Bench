from transformers import AutoImageProcessor, AutoModel

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('aimv2-huge-patch14-336')      
class AIMv2VisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower =  AutoModel.from_pretrained( cfg.model_name_or_path,trust_remote_code=True)
        self._image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
  
