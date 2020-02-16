"""
Import Libraries
"""
import argparse
import json
import PIL
import torch
import numpy as np

from math import ceil
from train import check_gpu
from torchvision import models, transforms

"""
Functions
"""

# Function arg_parser() to accept arguments from the command line
def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    # Point towards image for prediction
    parser.add_argument('--image', 
                        type=str, 
                        help='Point to impage file for prediction.',
                        required=True)

    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.',
                        required=True)
    
    # Specify top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.')
    
    # Import category names
    parser.add_argument('--category_names', 
                        type=str, 
                        help='Mapping from categories to real names.')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')

    # Parse args
    args = parser.parse_args()
    
    return args

# Function load_checkpoint(checkpoint_path) loads our saved deep learning model from checkpoint
def load_checkpoint(checkpoint_path):
    # Load the saved file
    checkpoint = torch.load("checkpoint.pth")
    
    # Load Defaults if none specified
    if checkpoint['architecture'] == 'resnet101':
        model = models.resnet101(pretrained=True)
        model.name = "resnet101"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Function process_image(image_path) performs cropping, scaling of image for our model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Get image
    image = Image.open(image_path)
    # TODO: Process a PIL image for use in a PyTorch model
    
    processed_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    image = processed_img(image)
    image = image.numpy()
    return image



def predict(image_path, model, cat_to_name, top_k):
    
    # check top_k
    if type(top_k) == type(None):
        top_k = 5
        print("Top K not specified, assuming K=5.")
    
    # Set model to evaluate
    model.eval()
    
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    model = model.cpu()
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_k)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    
    return top_probs, top_labels, top_flowers

# Converts two lists into a dictionary to print on screen
def print_probability(probs, flowers):
    
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))
        

"""
Main Function
"""

def main():
    
    # Get Keyword Args for Prediction
    args = arg_parser()
    
    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # Load model trained with train.py
    model = save_checkpoint(args.checkpoint)
    
    # Process Image
    image_tensor = process_image(args.image)
    
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Use `processed_image` to predict the top K most likely classes
    top_probs, top_labels, top_flowers = predict(image_tensor, model, 
                                                 cat_to_name,
                                                 args.top_k)
    
    # Print out probabilities
    print_probability(top_flowers, top_probs)
    
"""
RUN PROGRAM
"""
if __name__ == '__main__': main()
