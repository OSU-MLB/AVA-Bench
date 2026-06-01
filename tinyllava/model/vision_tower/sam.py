from . import register_vision_tower
from .base import VisionTower
import torch.nn.functional as F

from transformers import SamModel, SamProcessor

@register_vision_tower('sam')      
class SAMVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg) 
        self._vision_tower = SamModel.from_pretrained(cfg.model_name_or_path)
        self._image_processor = SamProcessor.from_pretrained(cfg.model_name_or_path)
        self._image_processor.crop_size = getattr(self._image_processor.image_processor, 'crop_size', {'height':1024,'width':1024})
        self._image_processor.size = getattr(self._image_processor.image_processor, 'size', None)

    def interpolate(self, image_features):
        target_h = target_w = 24
        image_features_flatten = F.interpolate(
            image_features.float(),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).to(dtype=image_features.dtype)
        image_features_flatten = image_features_flatten.flatten(2, 3).permute(0, 2, 1)
        return image_features_flatten.contiguous()
    
    def forward(self, x, **kwargs):
        image_features = self._vision_tower.vision_encoder(x,output_hidden_states=True)
        image_features = image_features['hidden_states'][-1].permute(0, 3, 1, 2).contiguous() #hidden states contain all the features Or returns after neck output(256 dim)
        interp_features= self.interpolate(image_features)
        return interp_features
