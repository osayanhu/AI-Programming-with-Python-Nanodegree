from torchvision import transforms,datasets,models
import torch
from torch import nn, optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse

#------------------------------------------------------------------------------------------------------

#--------------------------------------------
#
# Creating Argument Parser object named parser
#
#-----------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('image_path', type = str, 
                    help = 'Provide the path to a singe image (required)')
parser.add_argument('save_path', type = str, 
                    help = 'Provide the path to the file of the trained model (required)')

parser.add_argument('--category_names', type = str,default = 'cat_to_name.json'
                    help = 'Use a mapping of categories to real names')
parser.add_argument('--top_k', type = int, default = 5,
                    help = 'Return top K most likely classes. Default value is 5')
# GPU
parser.add_argument('--gpu', action='store_true',
                    help = "Add to activate CUDA")

args_in = parser.parse_args()


if args_in.gpu:
    device = torch.device("cuda")
    print("****** CUDA activated ********************")
else:
    device = torch.device("cpu")
    
#------------------------------------------------------------
#                         
#           Loading the checkpoint 
#
#------------------------------------------------------------

def load_checkpoint(filepath):
    model = models.vgg16(pretrained = True)
                         
    for param in model.parameters():
        param.requires_grad = False
   

    checkpoint = torch.load(filepath)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']
    model.class_to_idx=checkpoint['class_to_idx']
    
    
    return model, epochs, learning_rate

model, epochs, learning_rate = load_checkpoint(args_in.save_path)



#------------------------------------------------------------
#                         
#           Image Pre-processing 
#
#------------------------------------------------------------

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])

    return np.array(transform(img))

#------------------------------------------------------------
#                         
#           Prediction
#------------------------------------------------------------

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0).to(device) 
   
    model.eval()
    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        prob, labels = ps.topk(topk, dim=1)
        
        class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = list()
    
        for label in labels.cpu().numpy()[0]:
            classes.append(class_to_idx_inv[label])
        
        return prob.cpu().numpy()[0], classes

probs, classes = predict(args_in.image_path, model, topk = args_in.top_k)

with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
names = [cat_to_name[key] for key in classes]

print(f'The Predicted class name is : {names[0]}')
print(f'The Predicted class probability is : {probs[0]}')

if args_in.category_names:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[key] for key in classes]
    print(f'Top {args_in.top_k} Probabilities for {args_in.image_path}')
    for i,j in zip(names, probs):
        print(f'{i} : {j}')
    
     