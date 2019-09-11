import os
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
import torch
from Classes_Network import *
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import random
from torch import optim
import copy

ROOT_DIR = '../Dataset/'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_ANNO = 'Classes_train_annotation.csv'
VAL_ANNO = 'Classes_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']

class MyDataset():

    def __init__(self, root_dir, annotations_file, transform=None):

        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + 'does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None

        image = Image.open(image_path).convert('RGB')
        label_class = int(self.file_info.iloc[idx]['classes'])

        sample = {'image': image, 'classes': label_class}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample

train_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       ])
val_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor()
                                     ])

train_dataset = MyDataset(root_dir= ROOT_DIR + TRAIN_DIR,
                          annotations_file= TRAIN_ANNO,
                          transform=train_transforms)

test_dataset = MyDataset(root_dir= ROOT_DIR + VAL_DIR,
                         annotations_file= VAL_ANNO,
                         transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def visualize_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, CLASSES[sample['classes']])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()
#visualize_dataset()

def train_model(model, criterion, optimizer, num_epochs=50, lam=1.0):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_classes = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_classes = 0

            for idx,data in enumerate(data_loaders[phase]):
                #print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_classes = data['classes'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x_classes = model(inputs)

                    x_classes = x_classes.view(-1, 2)

                    _, preds_classes = torch.max(x_classes, 1)

                    reg = 0.
                    for param in model.parameters():
                        reg += torch.norm(param)
                    loss = criterion(x_classes, labels_classes) + lam * reg

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                corrects_classes += torch.sum(preds_classes == labels_classes)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc_classes

            Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
            print('{} Loss: {:.4f}  Acc_classes: {:.2%}'.format(phase, epoch_loss,epoch_acc_classes))

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc_classes
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val classes Acc: {:.2%}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pt')
    print('Best val classes Acc: {:.2%}'.format(best_acc))
    return model, Loss_list,Accuracy_list_classes

criterion = nn.CrossEntropyLoss()
for network_class in ["original","morelayer","withnorm"]:
    for method in ["SGD","Adam"]:
        if method == "SGD":
             lrs=[0.01,0.1,1.0]
        else:
             lrs=[1e-3,1e-4,1e-5]
        for lr in lrs:
            network = Net().to(device)
            if method == "SGD":
                optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9)
            else:
                optimizer = optim.Adam(network.parameters(), lr=lr)
            for lam in [0,0.1,1.0]:
                print(network_class,method,lr,lam)
                model, Loss_list, Accuracy_list_classes = train_model(network, criterion, optimizer, num_epochs=20, lam=lam)

                x = range(0, 20)
                y5 = Accuracy_list_classes["train"]
                y6 = Accuracy_list_classes["val"]
                plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
                plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
                plt.legend()
                plt.title('train and val Classes_acc vs. epoches')
                plt.ylabel('Classes_accuracy')
                plt.savefig("train and val Classes_acc vs epoches" + "Net_" + network_class + "_" + method + "_lr_" + str(lr) + "_lam_" + str(lam) + ".jpg")
                plt.close('all')

############################################ Visualization ###############################################
def visualize_model(model):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loaders['val']):
            inputs = data['image']
            labels_classes = data['classes'].to(device)

            x_classes = model(inputs.to(device))
            x_classes=x_classes.view( -1,2)
            _, preds_classes = torch.max(x_classes, 1)

            print(inputs.shape)
            plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
            plt.title('predicted classes: {}\n ground-truth classes:{}'.format(CLASSES[preds_classes],CLASSES[labels_classes]))
            plt.show()

#visualize_model(model)
