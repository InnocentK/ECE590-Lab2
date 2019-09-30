import argparse
import os, sys
import time
import datetime

# Import pytorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm

# You cannot change this line.
from tools.dataloader import CIFAR10, CIFAR100

# Create the neural network module: LeNet-5
class InnocentNet(nn.Module):
    def __init__(self):
        super(InnocentNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv1bn = nn.BatchNorm2d(6)

        self.conv12 = nn.Conv2d(6, 12, 3)
        self.conv12bn = nn.BatchNorm2d(12)

        self.conv13 = nn.Conv2d(12, 24, 3)
        self.conv13bn = nn.BatchNorm2d(24)

        self.conv14 = nn.Conv2d(24, 48, 3)
        self.conv14bn = nn.BatchNorm2d(48)

        self.conv15 = nn.Conv2d(48, 64, 3)
        self.conv15bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv2bn = nn.BatchNorm2d(128)
                
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv3bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 2)
        self.conv4bn = nn.BatchNorm2d(512)
       
        self.fc1 = nn.Linear(512*2*2, 1024)
        self.fc2 = nn.Linear(1024, 100)
        self.fc3 = nn.Linear(100, 10)
        
        
    def forward(self, x):
        out = F.relu(self.conv1bn(self.conv1(x)))
        out = F.relu(self.conv12bn(self.conv12(out)))
        #out = F.max_pool2d(out, 2)
        out = F.relu(self.conv13bn(self.conv13(out)))
        out = F.relu(self.conv14bn(self.conv14(out)))
        #out = F.max_pool2d(out, 2)
        out = F.relu(self.conv15bn(self.conv15(out)))
        #out = F.relu(self.conv12(out))
        out = F.max_pool2d(out, 2)
        #out = F.dropout2d(out, 0.05)

        out = F.relu(self.conv2bn(self.conv2(out)))
        #out = F.relu(self.conv22(out))
        out = F.max_pool2d(out, 2)
        out = F.dropout2d(out, 0.1)
        
        out = F.relu(self.conv3bn(self.conv3(out)))
        #out = F.relu(self.conv32(out))
        out = F.max_pool2d(out, 2)
        out = F.dropout2d(out, 0.25)
        
        out = F.relu(self.conv4bn(self.conv4(out)))
        #out = F.relu(self.conv42(out))
        out = F.max_pool2d(out, 2)
        out = F.dropout2d(out, 0.25)
        
        out = out.view(out.size(0), -1)
        
        out = F.relu(self.fc1(out))
        out = F.dropout2d(out, 0.5)
        
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Setting some hyperparameters
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 100
INITIAL_LR = 0.1#0.01
MOMENTUM = 0.85
REG = 1e-5
EPOCHS = 30
DATAROOT = "./data"
CHECKPOINT_PATH = "./saved_model"
OUTROOT = "./output"
NORMALIZE = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

def printOutput(epoch, val_acc, trial_no = 0):
    out_file = open(OUTROOT + "/results_" + str(trial_no) + ".csv", "a+")
    
    if epoch == 0:
        out_file.write("Epoch,Accuracy\n")

    out_file.write(str(epoch) + "," + str(val_acc) + "\n")
    out_file.close()

#def run(mytransform, trial, decay = 0.92, momentum = 0.85, epochs = 30):
#MOMENTUM = momentum
#REG = reg
#EPOCHS = epochs
decay = 0.92
trial = 100

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
#transform_train = mytransform

transform_val = transforms.Compose([
    transforms.ToTensor(), 
    NORMALIZE])

transform_test = transforms.Compose([
    transforms.ToTensor(), 
    NORMALIZE])

# Call the dataset Loader
trainset = CIFAR10(root=DATAROOT, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
valset = CIFAR10(root=DATAROOT, train=False, download=True, transform=transform_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

# Specify the device for computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = InnocentNet()
net = net.to(device)
if device =='cuda':
    print("Train on GPU...")
else:
    print("Train on CPU...")

# FLAG for loading the pretrained model
TRAIN_FROM_SCRATCH = False
# Code for loading checkpoint and recover epoch id.
CKPT_PATH = "./saved_model/model2.h5"
def get_checkpoint(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path)
    except Exception as e:
        print(e)
        return None
    return ckpt

ckpt = get_checkpoint(CKPT_PATH)
if ckpt is None or TRAIN_FROM_SCRATCH:
    if not TRAIN_FROM_SCRATCH:
        print("Checkpoint not found.")
    print("Training from scratch ...")
    start_epoch = 0
    current_learning_rate = INITIAL_LR
else:
    print("Successfully loaded checkpoint: %s" %CKPT_PATH)
    net.load_state_dict(ckpt['net'])
    start_epoch = ckpt['epoch'] + 1
    current_learning_rate = ckpt['lr']
    print("Starting from epoch %d " %start_epoch)

print("Starting from learning rate %f:" %current_learning_rate)

# Create loss function and specify regularization
criterion = nn.CrossEntropyLoss() #must add regularization
# Add optimizer
optimizer = optim.SGD(net.parameters(), lr= INITIAL_LR, momentum=MOMENTUM, weight_decay=REG)

# Start the training/validation process
# The process should take about 5 minutes on a GTX 1070-Ti
# if the code is written efficiently.
global_step = 0
best_val_acc = 0

for i in range(start_epoch, EPOCHS):
    print(datetime.datetime.now())
    # Switch to train mode
    net.train()
    print("Epoch %d:" %i)

    total_examples = 0
    correct_examples = 0

    train_loss = 0
    train_acc = 0
    # Train the training dataset for 1 epoch.
    print(len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Copy inputs to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        # Zero the gradient
        optimizer.zero_grad()
        # Generate output
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # Now backward loss
        loss.backward()
        # Apply gradient
        optimizer.step()
        # Calculate predicted labels
        _, predicted = outputs.max(1)
        # Calculate accuracy
        total_examples += 1
        correct_examples += (predicted == targets).sum()

        train_loss += loss

        global_step += 1
        if global_step % 100 == 0:
            avg_loss = train_loss / (batch_idx + 1)
        pass
    avg_acc = correct_examples / total_examples
    print("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, avg_acc))
    #printOutput(i, avg_acc.item(),trial)
    print(datetime.datetime.now())
    # Validate on the validation dataset
    print("Validation...")
    total_examples = 0
    correct_examples = 0
    
    net.eval()

    val_loss = 0
    val_acc = 0
    # Disable gradient during validation
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # Copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Zero the gradient
            optimizer.zero_grad()
            # Generate output from the DNN.
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # Calculate predicted labels
            _, predicted = outputs.max(1)
            # Calculate accuracy
            total_examples += 1#len(predicted)
            correct_examples += (predicted == targets).sum()#len([i for i in targets + predicted if i in targets and i in predicted])
            val_loss += loss

    avg_loss = val_loss / len(valloader)
    avg_acc = correct_examples / total_examples
    print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))
    printOutput(i, avg_acc.item(),trial)
        
    DECAY_EPOCHS = 2
    DECAY = decay#1.00
    if i % DECAY_EPOCHS == 0 and i != 0:
        current_learning_rate = INITIAL_LR * (DECAY**(EPOCHS // DECAY_EPOCHS)) #optim.lr_scheduler.StepLR(optimizer,step_size=DECAY_EPOCHS,gamma=DECAY)
        for param_group in optimizer.param_groups:
            # Assign the learning rate parameter
            param_group['lr'] = current_learning_rate

        print("Current learning rate has decayed to %f" %current_learning_rate)
    
    # Save for checkpoint
    if avg_acc > best_val_acc:
        best_val_acc = avg_acc
        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        print("Saving ...")
        state = {'net': net.state_dict(),
                'epoch': i,
                'lr': current_learning_rate}
        torch.save(state, os.path.join(CHECKPOINT_PATH, 'model2.h5'))

print("Optimization finished.")