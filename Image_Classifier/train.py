"""
Import Libraries
"""
import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from collections import OrderedDict
import copy

"""
Functions
""""
# Function arg_parser() to accept keyword arguments from the command line

def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    # Add architecture selection to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models as str')
    
    # Add checkpoint directory to parser
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints as str. If not specified then model will be lost.')
    
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate as float')
      
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for DNN classifier as int')

    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use Cuda for efficient calculations')
    
    # Parse args
    args = parser.parse_args()
    return args

# Function primaryloader_model(architecture="resnet101") downloads model (primary) from torchvision
def primaryloader_model(architecture="resnet101"):
    # Load Defaults if none specified
    if type(architecture) == type(None): 
        model = models.resnet101(pretrained=True)
        model.name = "resnet101"
        print("Network architecture specified as resnet101.")
    else: 
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    return model


# Function initial_classifier(model, hidden_units) creates a classifier 
def initial_classifier(model, hidden_units):
    
    # Check that hidden layers has been input
    if type(hidden_units) == type(None): 
        hidden_units = 512 #hyperparamters
        print("Number of Hidden Layers specificed as 4096.")
    
    # Find Input Layers
    input_features = model.fc[0].in_features
    
    # Define Classifier
    classifier = nn.Sequential(OrderedDict([
                          ('hd1', nn.Linear(input_features, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p = 0.2)),
                          ('hd2', nn.Linear(hidden_units, 102)),
                          ('out', nn.LogSoftmax(dim = 1))
                          ]))
    return classifier

# Function check_gpu(gpu_arg) make decision on using CUDA with GPU or CPU
def check_gpu(gpu_arg):
   # If gpu_arg is false then simply return the cpu device
    if not gpu_arg:
        return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

# Function train_model() for training
def train_model(model, criterion, scheduler, num_epochs = 20, device = 'cuda'):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        
        # Combining Training and Validation epoch
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            accuracy = 0
            
            # Iteration over dataset
            for images, labels in dataloaders[phase]:
                # Move tensors to GPU
                images, labels = images.to(device), labels.to(device)
                
                # Clear the gradients
                optimizer.zero_grad()
                
                # Forward pass
                # Track Gradients only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    
                    #Calculate Loss
                    outputs = model.forward(images)
                    loss = criterion(outputs, labels)
                    
                    # preds contains row tensor of best predicted class name of an image
                    _, preds = torch.max(outputs, 1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Calculate Running_loss and Accuracy
                running_loss += loss.item() * images.size(0)
                equality = preds == labels.data
                accuracy += torch.sum(equality)
                    
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / data_size[phase]
            epoch_acc = accuracy.double() / data_size[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
    print('Best Validation Accuracy: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model

#Function validate_model(model, testloader, device) to validate the trained model on test data images
def validate_model(model, testloader, device):
   # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))
    

def save_checkpoint(Model, Save_Dir, Train_data):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            # Create `class_to_idx` attribute in model
            Model.class_to_idx = Train_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {'architecture': Model.name,
                          'classifier': Model.fc,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            
            # Save checkpoint
            torch.save(checkpoint, 'checkpoint.pth')

        else: 
            print("Directory not found, model will not be saved.")
    

# Function main() which executes all of the functions above
def main():
    
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    dirs = {
        'train' : train_dir,
        'validation' : valid_dir,
        'test' : test_dir
    }
    
    #Transforms Dictionary
    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])]),

        'validation' : transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])]),

        'test' : transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])])
    }
    
    
    # Load the datasets with ImageFolder
    image_datasets = {x : datasets.ImageFolder(dirs[x], 
                                               transform = data_transforms[x]) for x in ['train', 'validation', 'test']}

    #Only Training Set has to be shuffled
    shuffle = {
        'train' : True,
        'validation' : False,
        'test' : False
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x : torch.utils.data.DataLoader(image_datasets[x], 
                                                   batch_size = 32, 
                                                   shuffle = shuffle[x]) for x in ['train', 'validation', 'test']}

    # Batch sizes of train, validation and test sets
    data_size = {x : len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    
    # Load Model
    model = primaryloader_model(architecture=args.arch)
    
    # Build Classifier
    model.fc = initial_classifier(model, hidden_units=args.hidden_units)
    
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: 
        learning_rate = args.learning_rate

    # Define loss, optimizer and scheduler
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)
    
    best_model = train_model(model, criterion, scheduler, 
                             num_epochs = 15, device = 'cuda')
    
    print("\nTraining process is now complete!!")
    
    # Quickly Validate the model
    validate_model(best_model, dataloaders['test'], device)
    
    # Save the model
    save_checkpoint(best_model, args.save_dir, image_datasets['train'])
    
    
"""
RUN THE PROGRAM
"""
if __name__ == '__main__': main()
