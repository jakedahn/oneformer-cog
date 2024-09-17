# This file was pulled from https://huggingface.co/docs/transformers/main/en/model_doc/oneformer#transformers.OneFormerModel.forward.example
import time
import torch
from PIL import Image
import requests
from transformers import OneFormerProcessor, OneFormerModel

# Start total timing
total_start_time = time.time()

# Download image
download_start_time = time.time()
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
download_end_time = time.time()

# Load processor and model
load_start_time = time.time()

processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_dinat_large", cache_dir="weights-cache")
model = OneFormerModel.from_pretrained("shi-labs/oneformer_ade20k_dinat_large", cache_dir="weights-cache")

# processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny", cache_dir="weights-cache")
# model = OneFormerModel.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny", cache_dir="weights-cache")
model.to('cuda')

load_end_time = time.time()

# Preprocess inputs
preprocess_start_time = time.time()
inputs = processor(image, ["semantic"], return_tensors="pt")
inputs.to('cuda')
preprocess_end_time = time.time()

# Inference
inference_start_time = time.time()
with torch.no_grad():
    outputs = model(**inputs)
inference_end_time = time.time()

# Extract predictions
mask_predictions = outputs.transformer_decoder_mask_predictions
class_predictions = outputs.transformer_decoder_class_predictions

# End total timing
total_end_time = time.time()

# Print shapes of predictions
print(f"👉 Mask Predictions Shape: {list(mask_predictions.shape)}, Class Predictions Shape: {list(class_predictions.shape)}")

# Print predictions
print(mask_predictions)
print(class_predictions)

# Print benchmarking results
print("\nPerformance Benchmarking Results:")
print(f"Image Download Time: {download_end_time - download_start_time:.4f} seconds")
print(f"Model Loading Time: {load_end_time - load_start_time:.4f} seconds")
print(f"Preprocessing Time: {preprocess_end_time - preprocess_start_time:.4f} seconds")
print(f"Inference Time: {inference_end_time - inference_start_time:.4f} seconds")
print(f"Total Execution Time: {total_end_time - total_start_time:.4f} seconds")
