import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class DHRNet(nn.Module):
    """
    DHRNet is a custom neural network model.

    Args:
        num_classes (int): The number of classes for classification.

    Attributes:
        num_classes (int): The number of classes for classification.
        conv1_1 (torch.nn.Conv2d): First convolutional layer.
        bn1_1 (torch.nn.BatchNorm2d): Batch normalization layer.
        conv1_2 (torch.nn.Conv2d): Second convolutional layer.
        bn1_2 (torch.nn.BatchNorm2d): Batch normalization layer.
        conv2_1 (torch.nn.Conv2d): Third convolutional layer.
        bn2_1 (torch.nn.BatchNorm2d): Batch normalization layer.
        conv2_2 (torch.nn.Conv2d): Fourth convolutional layer.
        bn2_2 (torch.nn.BatchNorm2d): Batch normalization layer.
        conv3_1 (torch.nn.Conv2d): Fifth convolutional layer.
        bn3_1 (torch.nn.BatchNorm2d): Batch normalization layer.
        conv3_2 (torch.nn.Conv2d): Sixth convolutional layer.
        bn3_2 (torch.nn.BatchNorm2d): Batch normalization layer.
        conv3_3 (torch.nn.Conv2d): Seventh convolutional layer.
        bn3_3 (torch.nn.BatchNorm2d): Batch normalization layer.
        btl1 (torch.nn.Conv2d): Bottleneck layer 1.
        btlu1 (torch.nn.Conv2d): Bottleneck layer 1 (upscaling).
        btl2 (torch.nn.Conv2d): Bottleneck layer 2.
        btlu2 (torch.nn.Conv2d): Bottleneck layer 2 (upscaling).
        btl3 (torch.nn.Conv2d): Bottleneck layer 3.
        btlu3 (torch.nn.Conv2d): Bottleneck layer 3 (upscaling).
        fc4 (torch.nn.Linear): Fully connected layer 4.
        fc5 (torch.nn.Linear): Fully connected layer 5.
        fc6 (torch.nn.Linear): Fully connected layer 6.
        deconv1 (torch.nn.ConvTranspose2d): First deconvolutional layer.
        deconv2 (torch.nn.ConvTranspose2d): Second deconvolutional layer.
        deconv3 (torch.nn.ConvTranspose2d): Third deconvolutional layer.

    """

    def __init__(self,num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.conv1_1 =  nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 =  nn.Conv2d(64, 64, kernel_size=3,
                               stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)


        self.conv2_1 =  nn.Conv2d(64, 128, kernel_size=3,
                               stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 =  nn.Conv2d(128, 128, kernel_size=3,
                               stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)


        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3,
                               stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3,
                               stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3,
                               stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)


        self.btl1 = nn.Conv2d(64, 32, kernel_size=3,
                               stride=1, padding=1)
        self.btlu1 = nn.Conv2d(32, 64, kernel_size=3,
                               stride=1, padding=1)
        self.btl2 = nn.Conv2d(128, 32, kernel_size=3,
                               stride=1, padding=1)
        self.btlu2 = nn.Conv2d(32, 128, kernel_size=3,
                               stride=1, padding=1)
        self.btl3 = nn.Conv2d(256, 32, kernel_size=3,
                               stride=1, padding=1)
        self.btlu3 = nn.Conv2d(32, 256, kernel_size=3,
                               stride=1, padding=1)


        self.fc4 = nn.Linear(4*4*256, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, self.num_classes)


        self.deconv1 = nn.ConvTranspose2d(64,3,kernel_size=2,stride=2,padding=0)
        self.deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,padding=0)
        self.deconv3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2,padding=0)
    
    def forward(self,x):
        """
        Forward pass of the DHRNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the output predictions, the output of the final deconvolutional layer, and a list of intermediate feature maps.

        """

        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
        x1 = F.max_pool2d(x1,kernel_size=2,stride=2)
        x1 = F.dropout(x1,p=0.25)

        x2 = F.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x2 = F.max_pool2d(x2,kernel_size=2,stride=2)
        x2 = F.dropout(x2,p=0.25)

        x3 = F.relu(self.bn3_1(self.conv3_1(x2)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = F.relu(self.bn3_3(self.conv3_3(x3)))
        x3 = F.max_pool2d(x3,kernel_size=2,stride=2)
        x3 = F.dropout(x3,p=0.25)
 
        x4 = x3.view(x3.size(0), -1)

        x5 = F.dropout(F.relu(self.fc4(x4)),p=0.5)
        x5 = F.dropout(F.relu(self.fc5(x5)),p=0.5)
        x5 = self.fc6(x5)


        z3 = F.relu(self.btl3(x3))  
        z2 = F.relu(self.btl2(x2))  
        z1 = F.relu(self.btl1(x1)) 

        j3 = self.btlu3(z3)
        j2 = self.btlu2(z2)
        j1 = self.btlu1(z1)


        g2 = F.relu(self.deconv3(j3))
        g1 = F.relu(self.deconv2(j2+g2))
        g0 = F.relu(self.deconv1(j1+g1))

        return x5, g0, [z3, z2, z1] #torch.concat([z3, z2, z1], dim = 0)
    
def epoch_train(net,device,trainloader,optimizer):
    """
    Performs a single epoch of training for the network.

    Args:
        net (torch.nn.Module): The neural network model.
        device (torch.device): The device on which the computations should be performed (e.g., 'cuda' for GPU or 'cpu' for CPU).
        trainloader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer for updating the network parameters.

    Returns:
        list: A list containing the training metrics: [accuracy, classification loss, reconstruction loss, total loss].

    """
        
    net.train() 
    correct=0
    total=0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iter=0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()

    for i,data in tqdm(enumerate(trainloader)):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits, reconstruct,_ = net(inputs)
        
        cls_loss = cls_criterion(logits, labels)

        reconst_loss = reconst_criterion(reconstruct,inputs)
      
        if(torch.isnan(cls_loss) or torch.isnan(reconst_loss)):
            print("Nan at iteration ",iter)
            cls_loss=0.0
            reconst_loss=0.0
            logits=0.0          
            reconstruct = 0.0  
            continue

        loss = cls_loss + reconst_loss

        loss.backward()
        optimizer.step()  

        total_loss = total_loss + loss.item()
        total_cls_loss = total_cls_loss + cls_loss.item()
        total_reconst_loss = total_reconst_loss + reconst_loss.item()

        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        iter = iter + 1

    return [(100 * (correct / total)), (total_cls_loss/iter), (total_reconst_loss/iter), (total_loss/iter)]

def epoch_val(net,device,testloader):
    """
    Performs a single epoch of validation for the network.

    Args:
        net (torch.nn.Module): The neural network model.
        device (torch.device): The device on which the computations should be performed (e.g., 'cuda' for GPU or 'cpu' for CPU).
        testloader (torch.utils.data.DataLoader): The test/validation data loader.

    Returns:
        list: A list containing the validation metrics: [accuracy, classification loss, reconstruction loss, total loss].

    """

    net.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iter=0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()

    with torch.no_grad():

      for data in testloader:

        images, labels = data[0].to(device), data[1].to(device)

        logits, reconstructs, _ = net(images)

        cls_loss = cls_criterion(logits, labels)

        reconst_loss = reconst_criterion(reconstructs,images)
    
        loss = cls_loss + reconst_loss

        total_loss = total_loss + loss.item()
        total_cls_loss = total_cls_loss + cls_loss.item()
        total_reconst_loss = total_reconst_loss + reconst_loss.item()

        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        iter = iter + 1

    return [(100 * (correct / total)), (total_cls_loss/iter), (total_reconst_loss/iter), (total_loss/iter)]

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    seed = 22
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    lr = 0.05
    epochs = 500
    batch_size = 128
    momentum= 0.9
    weight_decay= 0.0005
    means = [0.4914, 0.4822, 0.4465]
    stds = [0.2023, 0.1994, 0.2010]

    num_classes = 10
    print("Num classes "+str(num_classes))

    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.RandomAffine(degrees=30,translate =(0.2,0.2),scale=(0.75,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means,stds),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means,stds),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )

    net = DHRNet(num_classes)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=lr, 
                        momentum=momentum,weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(epochs):  # loop over the dataset multiple times

        train_acc = epoch_train(net,device,trainloader,optimizer)
        test_acc = epoch_val(net,device,testloader)
        scheduler.step()
        print("Train accuracy and cls, reconstruct and total loss for epoch "+
            str(epoch)+" is "+str(train_acc))       
        print("Test accuracy and cls, reconstruct and total loss for epoch "+
            str(epoch)+" is "+str(test_acc))

        torch.save(
            {
            'epoch':epoch,
            'model_state_dict':net.state_dict(),
            'train_acc':train_acc[0],
            'train_loss':train_acc[3],
            'val_acc':test_acc[0] ,
            'val_loss':test_acc[3]
            },
            "./model.pt")