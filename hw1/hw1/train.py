import numpy
import torch
import torch.optim as optim
import torch.nn as nn
from net import Net
from dataloader import train_dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import Adder

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    net = Net()
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    trainloader = train_dataloader("./dataset/", batch_size=16)
    max_iter = len(trainloader)
    writer = SummaryWriter()   
    epochs = 1000
    running_loss = Adder()
    
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss(loss.item())
            if i % 10 == 0:
                writer.add_scalar("Loss", running_loss.average() ,i + (epoch-1)* max_iter)
                first_output = outputs[0]
                list = first_output.tolist()
                maxidx = numpy.argmax(list)
                if maxidx==0:
                    writer.add_image("Light_Rain",inputs[0], global_step=i-1+ (epoch-1)* max_iter)
                elif maxidx==1:
                    writer.add_image("Heavy_Rain",inputs[0], global_step=i-1+ (epoch-1)* max_iter)
    
    torch.save(net.state_dict(),"./final.pkl")

    print('Finished Training')


