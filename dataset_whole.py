import cv2
import torch
import torch.nn as nn
import torchvision
import os
from torch.utils.data import Dataset,DataLoader  ##导入Dataset类

#用于整个模型训练的dataset
class DatasetWhole(Dataset):
    def __init__(self, root_dir,transform):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.img_paths = self._load_img_paths()
        self.transform = transform


    def _load_img_paths(self):
        img_paths = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for root, dirs, files in os.walk(class_dir):
                for img_name in files:
                    img_path = os.path.join(root, img_name)
                    img_paths.append((img_path, self.class_to_idx[class_name]))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

if __name__ == "__main__":
    dataloader=DataLoader()
    dataset = DatasetWhole(root_dir="data1")
    print("Number of images:", len(dataset))
    img, label = dataset[0]
    print("Label:", label)
    print("Image shape:", img.shape)