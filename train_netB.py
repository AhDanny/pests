import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_netB import DatasetNetB
import tensorboard
from resnetB import resnetB
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
import os
import datetime


batch_size=256
dataset_name='data1'
img_size=195
input_size=190
epoch_num=80
device='cuda:0'
file_path=f"model_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pth"#以开始训练的时间作为模型文件名字
tag=f"model_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"#以开始训练的时间作为模型在tensorboard的名字

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)  # 在第一个维度上取最大值的索引作为预测结果
    correct = (predicted == labels).sum().item()  # 计算预测正确的样本数量
    total = labels.size(0)  # 总样本数量
    accuracy = correct / total  # 计算精确度
    return accuracy

# 数据预处理方式
data_transforms = {
    'train': transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 数据集路径
train_root_path = os.path.join(os.getcwd(), dataset_name, 'train')
valid_root_path = os.path.join(os.getcwd(), dataset_name, 'val')


# 创建自定义数据集实例
train_dataset = DatasetNetB(root_dir=train_root_path, transform=data_transforms['train'])
valid_dataset = DatasetNetB(root_dir=valid_root_path, transform=data_transforms['val'])

# 创建数据加载器
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model=resnetB(4)
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
        x=x.to(device)
        label=label.to(device).long()
        output = model(x)
        loss = crossentropyloss(output, label)
        optimizer.zero_grad()
        loss.backward()
        top1_acc = accuracy(output, label)
        optimizer.step()
        sum_loss += loss.item()
        sum_top1_acc += top1_acc
        if indx%100==0:
            print('# Epoch: {} index{}| Loss: {:.4f} | top1_acc: {:.4f}'.format(epoch,indx,
                                                                         sum_loss / len(train_dataset),
                                                                         sum_top1_acc / len(train_dataset)
                                                                         ))
    model.eval()
    for indx,(x,label) in enumerate(validloader):
        x=x.to(device)
        label=label.to(device).long()
        output = model(x)
        loss = crossentropyloss(output, label)
        top1_acc = accuracy(output, label)
        sum_loss += loss.item()
        sum_top1_acc += top1_acc
        if indx%100==0:
            print('# Epoch: {} index{}| ValLoss: {:.4f} | Val_acc: {:.4f}'.format(epoch,indx,
                                                                         sum_loss / len(valid_dataset),
                                                                         sum_top1_acc / len(valid_dataset)
                                                                         ))
    writer.add_scalars(tag,{'LOSS/Train_loss':float(sum_loss / len(valid_dataset)),'ACC/Train_acc':float(sum_top1_acc / len(valid_dataset)),
                           'LOSS/val_loss':float(sum_loss / len(valid_dataset)),'ACC/val_acc':float(sum_top1_acc / len(valid_dataset))},
                      (epoch + 1))
    torch.save(model.state_dict(), file_path)
    scheduler.step()