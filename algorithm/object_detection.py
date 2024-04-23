import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor
from models.object import Object, Objects

class ObjectDetection():
    def __init__(self) -> None:
        self.image_processor = None
        self.model = None
    def load_model(self):
        self.image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    def predict (self, image: Image.Image) -> Objects:
        if not self.image_processor or not self.model:
            raise RuntimeError("Model is not loaded")
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        objects : list[Object]=[]
        for score, label, box in zip(results['scores'], results["labels"], results['boxes']):
            if score > 0.7:
                box_values = box.tolist()
                label = self.model.config.id2label[label.item()]
                objects.append(Object(box=box_values, label=label))
        return Objects(objects=objects)
