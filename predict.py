import torch
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from cog import BasePredictor, Input, Path
from PIL import Image
import numpy as np

CACHE_DIR = './weights-cache'

class Predictor(BasePredictor):
    def setup(self):
        """
        Load the model into memory to make running multiple predictions efficient.
        """
        # Load processor and model
        self.processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_ade20k_dinat_large",
            cached_dir=CACHE_DIR
        )
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_dinat_large",
            cached_dir=CACHE_DIR
        )
        # we want this to run on a GPU
        self.device = torch.device('cuda')
        self.model.to(self.device)

    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> Path:
        # Load and preprocess image
        img = Image.open(str(image))

        # Prepare inputs
        inputs = self.processor(img, ["semantic"], return_tensors="pt").to(self.device)

        # Run prediction
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs to get the semantic segmentation map
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[img.size[::-1]]
        )[0]

        # Move to CPU and convert to NumPy array
        predicted_semantic_map = predicted_semantic_map.cpu().numpy()

        # Find the label with the largest area
        labels, areas = np.unique(predicted_semantic_map, return_counts=True)
        sorted_idxs = np.argsort(-areas)
        labels = labels[sorted_idxs]
        label = labels[0]

        # Create binary mask
        binary_mask = (predicted_semantic_map == label).astype(np.uint8) * 255

        # Convert to image and save
        out_img = Image.fromarray(binary_mask)
        out_path = Path("output.png")
        out_img.save(out_path)

        return out_path


