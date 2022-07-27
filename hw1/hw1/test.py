import torch
from dataloader import test_dataloader
from utils import Adder
from net import Net
import torch.nn as nn

if __name__ == "__main__":
    batch_size = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testloader = test_dataloader("./dataset/", batch_size)
    
    net = Net()
    net.load_state_dict(torch.load("./final.pkl"))
    net.to(device)
    
    total = len(testloader) * batch_size
    correct = 0
    
    for i, data in enumerate(testloader):
        input_img, name = data
        input_img = input_img.to(device)
        # predict
        predict_label = net(input_img)
        _, predicted = torch.max(predict_label, 1)
        for j in range(len(name)):
            
            if predicted[j].item()==1 and name[j][:1] == 'h':
                correct+=1
            elif predicted[j].item()==0 and name[j][:1] == 'l':
                correct+=1
        
    print('Accuracy of the network on the 40 test images: %d %%' % ( 100 * correct / total))