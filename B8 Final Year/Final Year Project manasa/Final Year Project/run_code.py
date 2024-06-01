import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
# Paths where the model and processor were saved
model_save_directory = "./model"
processor_save_directory = "./processor"

# Load the model and processor from the current directory
model = CLIPModel.from_pretrained(model_save_directory)
processor = CLIPProcessor.from_pretrained(processor_save_directory)

def predict_violence(image, processor, model):
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Process the image and predict with CLIP
    inputs = processor(text=["violent scene", "non-violent scene"], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    violent_prob = probs[0][0].item()
    print(violent_prob)
    return violent_prob  # True if the scene is likely violent

def process_video(video_path, scale_factor=0.5):
    is_violent =False
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3) * scale_factor)
    frame_height = int(cap.get(4) * scale_factor)
    
    out = cv2.VideoWriter('processed_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        
        # Predict violence on the frame
        violent_prob = predict_violence(frame, processor, model)
                
        if violent_prob > 0.75:
            border_color = (0, 0, 255)  # Red in BGR
            is_violent = True
        elif violent_prob > 0.4:
            border_color = (0, 165, 255)  # Orange in BGR
        else:
            border_color = (0, 255, 0)  # Green in BGR

        bordered_frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
        
        out.write(bordered_frame)
        cv2.imshow('Frame', bordered_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    if is_violent:
        print("violent")

# Usage example
video_path = 'V_1.mp4'  # Replace with your local video file path
process_video(video_path)