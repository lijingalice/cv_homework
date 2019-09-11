import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from Multi_Network import *
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import random
from torch import optim
from torch.optim import lr_scheduler
import copy

ROOT_DIR = '../Dataset/'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_ANNO = 'Multi_train_annotation.csv'
VAL_ANNO = 'Multi_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']

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
        label_classes = int(self.file_info.iloc[idx]['classes'])
        label_species = int(self.file_info.iloc[idx]['species'])

        sample = {'image': image, 'classes': label_classes, 'species': label_species}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample


def getloader():
    """ generate data loader for both train and validation sets """
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

    return data_loaders

def visualize_dataset(train_loader):
    """ if you want to see some dataset, type visualize_dataset(data_loaders['train']) in IPython """
    idx = random.randint(0, len(train_loader.dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, CLASSES[sample['classes']], SPECIES[sample['species']])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()

def train_model(model, data_loaders, method="average", num_epoches = 20, 
                save_model_name = "best_model_weight1.5_lr0.01.pt",
                save_fig_name = "accuracy.vs.epoch_weight1.5_lr0.01.jpg"):
    """ only accuracy is stored for simplicity """
    Accuracy_list_classes = {'train': [], 'val': []}
    Accuracy_list_species = {'train': [], 'val': []}
   
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion_classes = nn.CrossEntropyLoss(weight=torch.tensor([1.0,2.0]))
    criterion_species = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch, num_epoches-1)) 
        print('-*'*10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            correct_classes = 0
            correct_species = 0

            for idx,data in enumerate(data_loaders[phase]):
                inputs = data['image']
                label_classes = data['classes']
                label_species = data['species']
                optimizer.zero_grad()
                weight = torch.ones(len(label_classes))
                
                with torch.set_grad_enabled(phase == 'train'):
                    x_classes, x_species = model(inputs)
                    x_classes = x_classes.view(-1,2)
                    x_species = x_species.view(-1,3)

                    _, preds_classes = torch.max(x_classes, 1)                    
                    _, preds_species = torch.max(x_species, 1)                    

                    correct_classes += torch.sum(preds_classes == label_classes)
                    correct_species += torch.sum(preds_species == label_species)
                    if phase == 'val':
                        print(list(zip(preds_classes, label_classes)))

                    if phase == 'train':
                        loss1 = criterion_classes(x_classes, label_classes)
                        loss2 = criterion_species(x_species, label_species)
                        #loss = loss1 + loss2 * 1.5
                        #loss = loss1
                        loss = loss1 + loss2
                        print("Epoch {}, phase {}, batch {}, loss={} {}".format(epoch,phase,idx,loss1,loss2))
                        loss.backward()
                        optimizer.step()
                        #scheduler.step()

            correct_classes = 100 * correct_classes.double()/len(data_loaders[phase].dataset)
            correct_species = 100 * correct_species.double()/len(data_loaders[phase].dataset)
            Accuracy_list_classes[phase].append(correct_classes)
            Accuracy_list_species[phase].append(correct_species)
            print("Epoch {}: phase={} classes acc={} species acc={}".
                  format(epoch, phase, correct_classes, correct_species,loss))
            epoch_acc = correct_classes + correct_species

            if phase == 'val' and (epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    model.load_state_dict(best_model_wts)
    torch.save(model,save_model_name)

    # save fig
    x = range(num_epoches)
    plt.plot(x, Accuracy_list_classes['train'], linestyle="-", marker=".", linewidth=1, label="train classes")
    plt.plot(x, Accuracy_list_classes['val'], linestyle="-", marker=".", linewidth=1, label="val classes")
    plt.plot(x, Accuracy_list_species['train'], linestyle="-", marker=".", linewidth=1, label="train species")
    plt.plot(x, Accuracy_list_species['val'], linestyle="-", marker=".", linewidth=1, label="val species")
    plt.legend()
    plt.savefig(save_fig_name)
    plt.close('all')

def train_model_twostep(model, data_loaders, num_epoches = 20, num_epoches1 = 12, 
                save_model_name = "best_model_twostep_12_8.pt",
                save_fig_name = "accuracy.vs.epoch_twostep_12_8.jpg"):
    """ only accuracy is stored for simplicity """
    """ this does species first for num_epoces1 then classes """
    Accuracy_list_classes = {'train': [], 'val': []}
    Accuracy_list_species = {'train': [], 'val': []}
   
    criterion_classes = nn.CrossEntropyLoss(weight=torch.tensor([1.0,2.0]))
    criterion_species = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch, num_epoches-1)) 
        print('-*'*10)

        # I am surprised that I need to redefine the optimizer
        if (epoch == 0):
            model.fc2_classes.weight.requires_grad = False            
            model.fc2_classes.bias.requires_grad = False            
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            #scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        if (epoch == num_epoches1):
            for para in model.parameters():
                para.requires_grad = False
            model.fc2_classes.weight.requires_grad = True 
            model.fc2_classes.bias.requires_grad = True           
            optimizer = optim.SGD(model.fc2_classes.parameters(), lr=0.01, momentum=0.9)
            #scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                # this doesn't work!!!
                #if epoch < num_epoches1:  # this train all layers except fc2_classes
                #    model.fc2_classes.weight.requires_grad = False
                #    model.fc2_classes.bias.requires_grad = False        
                #else:    # this only trains fc2_classes
                #    for para in model.parameters():
                #        para.requires_grad = False
                #    model.fc2_classes.weight.requires_grad = True 
                #    model.fc2_classes.bias.requires_grad = True           
            else:
                model.eval()
            for layer in model.children():
                print(layer)
                print(list(layer.parameters()))

            correct_classes = 0
            correct_species = 0

            for idx,data in enumerate(data_loaders[phase]):
                inputs = data['image']
                label_classes = data['classes']
                label_species = data['species']
                optimizer.zero_grad()
                weight = torch.ones(len(label_classes))
                
                with torch.set_grad_enabled(phase == 'train'):
                    x_classes, x_species = model(inputs)
                    x_classes = x_classes.view(-1,2)
                    x_species = x_species.view(-1,3)

                    _, preds_classes = torch.max(x_classes, 1)                    
                    _, preds_species = torch.max(x_species, 1)                    

                    correct_classes += torch.sum(preds_classes == label_classes)
                    correct_species += torch.sum(preds_species == label_species)
                    if phase == 'val':
                        print(list(zip(preds_classes, label_classes)))

                    if phase == 'train':
                        # this shows two losses, we only need one if we want faster speed
                        loss1 = criterion_classes(x_classes, label_classes)
                        loss2 = criterion_species(x_species, label_species)
                        if epoch < num_epoches1:
                            loss = loss2
                        else:
                            loss = loss1
                        print("Epoch {}, phase {}, batch {}, loss={} {}".format(epoch,phase,idx,loss1,loss2))
                        loss.backward()
                        optimizer.step()
                        #scheduler.step()

            correct_classes = 100 * correct_classes.double()/len(data_loaders[phase].dataset)
            correct_species = 100 * correct_species.double()/len(data_loaders[phase].dataset)
            Accuracy_list_classes[phase].append(correct_classes)
            Accuracy_list_species[phase].append(correct_species)
            print("Epoch {}: phase={} classes acc={} species acc={}".
                  format(epoch, phase, correct_classes, correct_species,loss))
            epoch_acc = correct_classes + correct_species

            if phase == 'val' and (epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    model.load_state_dict(best_model_wts)
    torch.save(model,save_model_name)

    # save fig
    x = range(num_epoches)
    plt.plot(x, Accuracy_list_classes['train'], linestyle="-", marker=".", linewidth=1, label="train classes")
    plt.plot(x, Accuracy_list_classes['val'], linestyle="-", marker=".", linewidth=1, label="val classes")
    plt.plot(x, Accuracy_list_species['train'], linestyle="-", marker=".", linewidth=1, label="train species")
    plt.plot(x, Accuracy_list_species['val'], linestyle="-", marker=".", linewidth=1, label="val species")
    plt.legend()
    plt.savefig(save_fig_name)
    plt.close('all')


if (__name__ == "__main__"):
    data_loaders = getloader()
    #train_model(model, data_loaders)

    #print("doing 10-10")
    #model = Net() 
    #train_model_twostep(model, data_loaders, num_epoches1 = 10, save_model_name = "best_model_twostep_10_10.pt",
    #            save_fig_name = "accuracy.vs.epoch_twostep_10_10.jpg")
    
    #print("doing 12-8")
    #model = Net() 
    #train_model_twostep(model, data_loaders, num_epoches1 = 12, save_model_name = "best_model_twostep_12_8.pt",
    #            save_fig_name = "accuracy.vs.epoch_twostep_12_8.jpg")

    print("doing 15-5")
    model = Net() 
    train_model_twostep(model, data_loaders, num_epoches1 = 15, save_model_name = "best_model_twostep_15_5.pt",
                save_fig_name = "accuracy.vs.epoch_twostep_15_5.jpg")
