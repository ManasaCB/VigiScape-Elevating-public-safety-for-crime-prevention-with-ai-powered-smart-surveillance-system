from transformers import CLIPProcessor, CLIPModel

# Paths where the model and processor were saved
model_save_directory = "./model"
processor_save_directory = "./processor"

# Load the model and processor from the current directory
model = CLIPModel.from_pretrained(model_save_directory)
processor = CLIPProcessor.from_pretrained(processor_save_directory)


import torch
from PIL import Image

def predict_violence(image, processor, model):
    # Process the image
    inputs = processor(text=["violent scene", "non-violent scene"], images=image, return_tensors="pt", padding=True)

    # Predict with CLIP
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

    # Assuming the first label is "violent scene" and the second is "non-violent scene"
    violent_prob = probs[0][0].item()

    return violent_prob
