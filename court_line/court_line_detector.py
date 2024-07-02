import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np
import json

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()  
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints.tolist()  # Convert to list for JSON serialization

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def process_video(self, video_frames, use_saved_keypoints=None, keypoints_path=None):
        if use_saved_keypoints and keypoints_path:
            keypoints_list = self.load_keypoints(keypoints_path)
        else:
            keypoints_list = []
            for frame in video_frames:
                keypoints = self.predict(frame)
                keypoints_list.append(keypoints)
            if keypoints_path:
                self.save_keypoints(keypoints_list, keypoints_path)
        
        output_video_frames = self.draw_keypoints_on_video(video_frames, keypoints_list)
        return output_video_frames

    def save_keypoints(self, keypoints_list, file_path):
        with open(file_path, 'w') as f:
            json.dump(keypoints_list, f)
    
    def load_keypoints(self, file_path):
        with open(file_path, 'r') as f:
            keypoints_list = json.load(f)
        return keypoints_list
    
    def draw_keypoints_on_video(self, video_frames, keypoints_list):
        output_video_frames = []
        for frame, keypoints in zip(video_frames, keypoints_list):
            frame_with_keypoints = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame_with_keypoints)
        return output_video_frames
