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

parser.add_argument('data_dir', type = str, #default = 'flowers',
                    help = 'Provide the data directory, mandatory')
parser.add_argument('--save_dir', type = str, default = './',
                    help = 'Provide the save directory')
# hyperparameters
parser.add_argument('--arch', type = str, default = 'vgg16',
                    help = 'densenet121 or vgg16')
parser.add_argument('--learning_rate', type = float, default = 0.003,
                    help = 'Learning rate, default value 0.001')
parser.add_argument('--hidden_units', type = int, default = 512,
                    help = 'First layers of hidden units. Default value is 512')
parser.add_argument('--hidden_units2', type = int, default = 256,
                    help = 'Second layers of hidden units. Default value is 256')
parser.add_argument('--epochs', type = int, default = 10,
                    help = 'Number of epochs')
# GPU
parser.add_argument('--gpu', action='store_true',
                    help = "Add to activate CUDA")

#setting values data loading
args_in = parser.parse_args()


if args_in.gpu:
    device = torch.device("cuda")
    print("****** CUDA activated ********************")
else:
    device = torch.device("cpu")
   
#---------------------------------------------------------------------


    
#--------------------------------------------
#
# Loading the Data
#
#-----------------------------------------------

  
data_dir = args_in.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Defining transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), 
                                                            (0.229, 0.224, 0.225))]) 

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), 
                                                            (0.229, 0.224, 0.225))])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), 
                                                            (0.229, 0.224, 0.225))])

# Loading the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
  
#----------------------------------------------------------------------------------



#--------------------------------------------
#
# Building the model
#
#-----------------------------------------------





layers = args_in.hidden_units
layers_2 = args_in.hidden_units2
lr = args_in.learning_rate
    
if args_in.arch == 'vgg16':
    model = models.vgg16(pretrained = True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
          nn.Linear(25088, layers),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(layers, layers_2),
          nn.SELU(),
          nn.Dropout(p=0.2),
          nn.Linear(layers_2, 102),
          nn.LogSoftmax(dim = 1)
        )
elif args_in.arch == 'densenet121:
    model = models.densenet121(pretrained = True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        model.classifier = nn.Sequential(
          nn.Linear(25088, layers),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(layers, layers_2),
          nn.SELU(),
          nn.Dropout(p=0.2),
          nn.Linear(layers_2, 102),
          nn.LogSoftmax(dim = 1)
        )
else:
    raise ValueError('Model arch error.')

model.classifier = classifier
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("------ model building finished -----------")

#---------------------------------------------------------------------------


#--------------------------------------------
#
# Training the model
#
#-----------------------------------------------

epochs = args_in.epochs
steps = 0
running_loss = 0
print_every = 15

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
     
#------------------------------------

#--------------------------------------------
#
# Testing the model
#
#-----------------------------------------------

test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test loss: {test_loss/len(testloader):.3f}.. "
      f"Test accuracy: {accuracy/len(testloader):.3f}")
running_loss = 0


#--------------------------------------------
#
# Save the checkpoint
#
#-----------------------------------------------

checkpoint = {
    'epochs': epochs,
    'learning_rate': 0.003,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion_state_dict': criterion.state_dict(),
    'class_to_idx': train_data.class_to_idx,
    'classifier':model.classifier
}

torch.save(checkpoint, args_in.save_dir+ 'checkpoint.pth')