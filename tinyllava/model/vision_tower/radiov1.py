from . import register_vision_tower
from .base import VisionTower
import torch.nn.functional as F

from transformers import AutoModel, CLIPImageProcessor

@register_vision_tower('radio')      
class RadioVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = AutoModel.from_pretrained(cfg.model_name_or_path,trust_remote_code=True)
        self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path,trust_remote_code=True)
        #self._image_processor.crop_size = {"height": 384, "width": 384}
        #self._image_processor.size =  {"shortest_edge": 384}
        self._image_processor.crop_size = {"height": 432, "width": 432}
        self._image_processor.size =  {"shortest_edge": 432}
        self._image_processor.do_resize = True
        self._image_processor.do_center_crop = True


    def forward(self, x,**kwargs):
        image_features = self._vision_tower.model.forward_intermediates(x,indices=[-2],intermediates_only=True,output_fmt='NLC')[0]
        #the model deletes the cls token when calling the above function. [bs,576,1024]
        return image_features
