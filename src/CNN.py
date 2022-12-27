import os
import time
import torch
import torch.nn as nn
from PIL import Image
from torchsummary import summary
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.pooling import MaxPool1d, MaxPool2d
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

def load_all_images(test_num, train_num, classes):
    test_path = './classed_dataset/test_data/'
    train_path = './classed_dataset/trainning_data/'
    # load test_data
    img_test = np.zeros([test_num,64,64], dtype='float32')
    test_labels = np.zeros(test_num,dtype='uint8')
    label_num = 0
    count = 0
    for i in classes:
        images = os.listdir(test_path+i)
        for j in images:
            image_path = test_path+i+'/'+j
            img = Image.open(image_path)
            np_img = np.array(img)
            img_test[count] = np_img/255
            test_labels[count] = label_num
            count+=1
        label_num+=1

    img_train = np.zeros([train_num,64,64], dtype='float32')
    train_labels = np.zeros(train_num,dtype='uint8')
    label_num = 0
    count = 0
    for i in classes:
        images = os.listdir(train_path+i)
        for j in images:
            image_path = train_path+i+'/'+j
            img = Image.open(image_path)
            np_img = np.array(img)
            img_train[count] = np_img/255
            train_labels[count] = label_num
            count+=1
        label_num+=1

    return img_test,test_labels,img_train,train_labels

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        return X

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), # [64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [64, 32, 32]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [128, 16, 16]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [256, 8, 8]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [512, 4, 4]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [512, 2, 2]
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 25)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1) 
        return self.fc(x)
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = os.listdir('./classed_dataset/test_data')
    test_num = 0
    for i in classes:
        images = os.listdir('./classed_dataset/test_data/'+i)
        test_num+=len(images)
    train_num = 0
    for i in classes:
        images = os.listdir('./classed_dataset/trainning_data/'+i)
        train_num+=len(images)

    img_test,test_labels,img_train,train_labels = load_all_images(test_num, train_num, classes)

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    batch_size = 4
    train_set = ImgDataset(img_train, train_labels, data_transform)
    test_set = ImgDataset(img_test, test_labels, data_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle = False)

    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
    
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  
            pred = model(data[0].to(device))
            batch_loss = criterion(pred, data[1]) 
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(pred.detach().numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                pred = model(data[0].to(device))
                batch_loss = criterion(pred, data[1]) 
                val_acc += np.sum(np.argmax(pred.detach().numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()
        
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch+1, epochs, time.time()-epoch_start_time, \
                train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/test_set.__len__(), val_loss/test_set.__len__()))

if __name__=='__main__':
    main()