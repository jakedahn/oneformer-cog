from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests
import torch

# load OneFormer fine-tuned on ADE20k for universal segmentation
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
model.to('cuda')

url = (
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
)
image = Image.open(requests.get(url, stream=True).raw)

# Semantic Segmentation
inputs = processor(image, ["semantic"], return_tensors="pt")
inputs.to('cuda')

with torch.no_grad():
    outputs = model(**inputs)
# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to processor for semantic postprocessing
predicted_semantic_map = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]
print(f"👉 Semantic Predictions Shape: {list(predicted_semantic_map.shape)}")

# Instance Segmentation
inputs = processor(image, ["instance"], return_tensors="pt")
inputs.to('cuda')
with torch.no_grad():
    outputs = model(**inputs)
# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to processor for instance postprocessing
predicted_instance_map = processor.post_process_instance_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]["segmentation"]
print(f"👉 Instance Predictions Shape: {list(predicted_instance_map.shape)}")

# Panoptic Segmentation
inputs = processor(image, ["panoptic"], return_tensors="pt")
inputs.to('cuda')
with torch.no_grad():
    outputs = model(**inputs)
# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to processor for panoptic postprocessing
predicted_panoptic_map = processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]["segmentation"]
print(f"👉 Panoptic Predictions Shape: {list(predicted_panoptic_map.shape)}")