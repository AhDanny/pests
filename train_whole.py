import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_whole import DatasetWhole
import tensorboard
from model import Model
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
import os
import datetime

numclass=4#标签的数量
batch_size=256
dataset_name='shujuji'#数据集的根目录
img_size=195
input_size=190
epoch_num=80
device='cpu'
file_path=f"model_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pth"#以开始训练的时间作为模型文件名字
tag=f"model_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"#以开始训练的时间作为模型在tensorboard的名字

# 数据预处理方式
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 数据集路径
train_root_path = os.path.join(os.getcwd(), dataset_name, 'train')
valid_root_path = os.path.join(os.getcwd(), dataset_name, 'valid')


# 创建自定义数据集实例
train_dataset = DatasetWhole(root=train_root_path, transform=data_transforms['train'])
valid_dataset = DatasetWhole(root=valid_root_path, transform=data_transforms['val'])

# 创建数据加载器
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

model=Model()
model.to(device)
crossentropyloss=CrossEntropyLoss().to(device=device)
writer = SummaryWriter("mylog")
initial_lr = 0.01
momentum = 0.9
weight_decay = 5e-4
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum,
                      weight_decay=weight_decay, nesterov=True)
scheduler = CosineAnnealingLR(optimizer,eta_min=0, T_max=epoch_num)

for epoch in range(epoch_num):
    sum_top1_acc = 0
    sum_loss = 0
    model.train()
    for indx,(x,label) in enumerate(trainloader):
        output = model(x)
        loss = crossentropyloss(output, label.float())
        optimizer.zero_grad()
        loss.backward()
        top1_acc = torch.eq(output, label).float().mean()
        optimizer.step()
        sum_loss += loss.item()
        sum_top1_acc += top1_acc
        if indx%100==0:
            print('# Epoch: {} index{}| Loss: {:.4f} | top1_acc: {:.4f}'.format(epoch,indx,
                                                                         sum_loss / len(train_dataset),
                                                                         sum_top1_acc / len(train_dataset)
                                                                         ))
    model.eval()
    for indx,(x,label) in enumerate(trainloader):
        output = model(x)
        loss = crossentropyloss(output, label.float())
        top1_acc = torch.eq(output, label).float().mean()
        sum_loss += loss.item()
        sum_top1_acc += top1_acc
        if indx%100==0:
            print('# Epoch: {} index{}| ValLoss: {:.4f} | Val_acc: {:.4f}'.format(epoch,indx,
                                                                         sum_loss / len(valid_dataset),
                                                                         sum_top1_acc / len(valid_dataset)
                                                                         ))
    writer.add_scalars(tag,{'LOSS/Train_loss':float(sum_loss / len(train_dataset)),'ACC/Train_acc':float(sum_top1_acc / len(train_dataset)),
                           'LOSS/val_loss':float(sum_loss / len(valid_dataset)),'ACC/val_acc':float(sum_top1_acc / len(valid_dataset))},
                      (epoch + 1))
    torch.save(model.state_dict(), file_path)
    scheduler.step()