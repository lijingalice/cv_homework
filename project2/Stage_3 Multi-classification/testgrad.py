import torch
from torch import optim
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(5,2)
        self.fc2 = nn.Linear(5,3)

    def forward(self,x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1,x2

def testgrad():
    x = torch.rand(3,5)
    x1_label = torch.tensor([0,0,1])
    x2_label = torch.tensor([0,1,2])
    model = Net()

    print("--- first round ---")
    x1,x2 = model(x)
    loss1 = nn.CrossEntropyLoss()(x1,x1_label) 
    loss1.backward()
    for param in model.parameters():
        #print("value:", param)
        print("grad:", param.grad)


    print("--- second round ---")
    model.fc1.bias.requires_grad = False
    x1,x2 = model(x)
    loss1 = nn.CrossEntropyLoss()(x1,x1_label)
    loss1.backward()
    print(" -- now notice that the grads have been accumulated, except the bias!!! --")
    for param in model.parameters():
        #print("value:", param)
        print("grad:", param.grad)

   
    print("--- third round ---")
    model.zero_grad()
    model.fc1.bias.requires_grad = False
    x1,x2 = model(x)
    loss1 = nn.CrossEntropyLoss()(x1,x1_label)
    loss1.backward()
    print(" -- after adding model.zero_grad() --- ")
    for param in model.parameters():
        #print("value:", param)
        print("grad:", param.grad) 

 
    print("--- third round ---")
    model=Net()
    print(" -- raw weight --")
    for param in model.parameters():
        print("value:",param)
    optimizer = optim.SGD(model.parameters(),lr=10.0,momentum=0.9)
    x1,x2 = model(x)
    loss1 = nn.CrossEntropyLoss()(x1,x1_label)
    loss1.backward()
    optimizer.step() 
    print(" -- taking one step -- ")
    for param in model.parameters():
        print("value:",param)
    optimizer.zero_grad()
    print(" -- taking one more step after zero_grad!!! -- ")
    optimizer.step()
    for param in model.parameters():
        print("value:",param)
    print(" -- even the grads are zeros!!! -- ")
    for param in model.parameters():
        print("grad:",param.grad)
    print(" -- and it's because the momentum! note the change changes by 0.9 -- ")
    optimizer.step()
    for param in model.parameters():
        print("value:",param)
    
