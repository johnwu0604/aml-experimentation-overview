import json
import base64
import os
import pickle
import torch
import torch.nn as nn
from torchvision import transforms
from azureml.core.model import Model
from io import BytesIO
from PIL import Image

def preprocess_image(image_file):
    """Preprocess the input image."""
    data_transforms = transforms.Compose([
        transforms.Resize(200),
        transforms.CenterCrop(200),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_file)
    image = data_transforms(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image


def init():
    global model, classes
    model_path = Model.get_model_path('cifar-classifier')
    model = torch.load(os.path.join(model_path,'model.pt'), map_location=torch.device('cpu'))
    model.eval()
    pkl_file = open(os.path.join(model_path,'class_names.pkl'), 'rb')
    classes = pickle.load(pkl_file)
    pkl_file.close()    

def run(input_data):
    
    results = []
    
    for image_url in input_data:
        # preprocess image
        img = preprocess_image(image_url)
        # get prediction
        output = model(img)
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(model(img)).detach().numpy()[0]
        index = torch.argmax(output, 1)
        # add to results
        results.append('{}: {}'.format(image_url, classes[index]))
        
    return results
