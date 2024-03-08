import cv2
import torch
import torch.nn as nn
import torchvision
import os
from torch.utils.data import Dataset,DataLoader  ##导入Dataset类

#用于训练网路B的dataset

class DatasetNetB(Dataset):
    def __init__(self, root_dir,transform):
        super().__init__()
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.labels = {'egg':0,'larva':1,'pupa':2,'imago':3}
        self.img_paths = self._load_img_paths()
        self.transform = transform


    def _load_img_paths(self):
        img_paths = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, file)
                img_paths.append((img_path,self.labels[class_name]))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = torch.FloatTensor(img).reshape(3,img.shape[0],img.shape[1])
        if self.transform:
            img = self.transform(img)
        return img, label

if __name__ == "__main__":
    dataset = DatasetNetB(root_dir="data1")
    print("Number of images:", len(dataset))
    img, label = dataset[0]
    print("Label:", label)
    print("Image shape:", img.shape)