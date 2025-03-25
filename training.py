import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from model import Myresnet18
class GestureDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        label = self.label_list[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class Training():
    def __init__(self, batch_size, epochs, categories, train_folder, test_folder, model_name, type):
        self.batch_size = batch_size
        self.epochs = epochs
        self.categories = categories
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.model_name = model_name
        self.type = type
        self.shape1 = 224
        self.shape2 = 224

    def read_train_images(self, folder):
        """从文件夹中读取图像和标签，放回图像列表和标签列表"""
        img_list = []
        label_list = []
        for file in os.listdir(folder):
            img = Image.open(folder + file)
            img = img.convert("L").resize((self.shape1, self.shape2))  # Convert to grayscale and resize
            img_list.append(folder + file)
            label_list.append(int(file.split('_')[1][0]))  
        return img_list, label_list

    def train(self):
        train_img_list, train_label_list = self.read_train_images(folder=self.train_folder)
        test_img_list, test_label_list = self.read_train_images(folder=self.test_folder)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        # Create datasets and dataloaders
        train_dataset = GestureDataset(train_img_list, train_label_list, transform)
        test_dataset = GestureDataset(test_img_list, test_label_list, transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = Myresnet18(self.categories).cuda()  # Move model to GPU
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Evaluate the model
        self.evaluate(model, test_loader)

        # Save the model
        torch.save(model, self.model_name)

    def evaluate(self, model, test_loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        confusion_mat = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(confusion_mat)

    def plot_confusion_matrix(self, cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.figure(figsize=(7, 5))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(self.categories)
        plt.xticks(tick_marks, self.type, rotation=45)
        plt.yticks(tick_marks, self.type)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

if __name__ == "__main__":
    train = Training(batch_size=32, epochs=20, categories=4,
                     train_folder='Gesture_train121/', test_folder='Gesture_predict121/',
                     model_name='gesture_model.pth', type=['hello', 'ok', 'you','to'])
    train.train()
