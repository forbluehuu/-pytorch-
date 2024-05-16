import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from torch.autograd import Variable
from tqdm import tqdm

import os
from PIL import Image
import cv2


#判断环境是CPU运行还是GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE


transform111 = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
transform111


import shutil
modellr = 1e-3
BATCH_SIZE = 64
EPOCHS = 100

# 删除隐藏文件/文件夹
for root, dirs, files in os.walk('./dataset'):
    for file in files:
        if 'ipynb_checkpoints' in file:
            os.remove(os.path.join(root, file))
    if 'ipynb_checkpoints' in root:
        shutil.rmtree(root)

# 读取数据
dataset_train = datasets.ImageFolder('D:/大学/人工智能/.kaggle/dataset/train', transform111)
print(dataset_train.imgs)
# 对应文件夹的label
print(dataset_train.class_to_idx)
dataset_test = datasets.ImageFolder('D:/大学/人工智能/.kaggle/dataset/val', transform111)
# 对应文件夹的label
print(dataset_test.class_to_idx)

dataset = 'D:\大学\人工智能\.kaggle\dataset'
train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'val')

batch_size = 32
num_classes = 6
print(train_directory)
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=transform111['train']),
    'val': datasets.ImageFolder(root=valid_directory, transform=transform111['val'])
}


train_data_size = len(data['train'])
valid_data_size = len(data['val'])

train_loader = torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(data['val'], batch_size=batch_size, shuffle=True, num_workers=8)

print(train_data_size, valid_data_size)



# 下载预训练模型
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)  # 不使用训练好的预训练模型
model


# 判断环境是CPU还是GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据预处理
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# 设置超参数
modellr = 1e-3
BATCH_SIZE = 64
EPOCHS = 100

# 删除隐藏文件/文件夹
for root, dirs, files in os.walk('./dataset'):
    for file in files:
        if 'ipynb_checkpoints' in file:
            os.remove(os.path.join(root, file))
    if 'ipynb_checkpoints' in root:
        shutil.rmtree(root)

# 读取数据
dataset_train = datasets.ImageFolder('D:/大学/人工智能/.kaggle/dataset/train', transform['train'])
dataset_test = datasets.ImageFolder('D:/大学/人工智能/.kaggle/dataset/val', transform['val'])

# 加载数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# 定义模型
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 二分类问题，将全连接层的输出改为2
model = model.to(DEVICE)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=modellr)

# 判断环境是CPU还是GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据预处理
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# 设置超参数
modellr = 1e-3
BATCH_SIZE = 64
EPOCHS = 100

# 删除隐藏文件/文件夹
for root, dirs, files in os.walk('./dataset'):
    for file in files:
        if 'ipynb_checkpoints' in file:
            os.remove(os.path.join(root, file))
    if 'ipynb_checkpoints' in root:
        shutil.rmtree(root)

# 读取数据
dataset_train = datasets.ImageFolder('D:/大学/人工智能/.kaggle/dataset/train', transform['train'])
dataset_test = datasets.ImageFolder('D:/大学/人工智能/.kaggle/dataset/val', transform['val'])

# 加载数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# 定义模型
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)  # 不使用训练好的预训练模型
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 二分类问题，将全连接层的输出改为2
model = model.to(DEVICE)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=modellr)


# 训练过程
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    return train_loss


# 验证过程
def validate(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss = running_loss / len(test_loader)
    val_accuracy = (100.0 * correct) / total
    return val_loss, val_accuracy


# 开始训练
best_val_loss = float('inf')
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    train_loss = train(model, DEVICE, train_loader, optimizer, criterion)
    val_loss, val_accuracy = validate(model, DEVICE, test_loader, criterion)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')

    print('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(
        epoch + 1, train_loss, val_loss, val_accuracy))


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=modellr)


def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 计算训练损失
        train_loss += loss.item()

        # 计算训练准确率
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss /= len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc


def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 计算测试损失
            test_loss += loss.item()

            # 计算测试准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct / total

    return test_loss, test_acc


train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch + 1}/{EPOCHS}:")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 定义模型
model = torchvision.models.resnet50(pretrained=True)  # 使用预训练的 ResNet50 模型
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 替换最后一层全连接层，输出为2，分别代表猫和狗
model = model.to(DEVICE)  # 将模型移动到设备上

# 加载预训练模型的权重
model.load_state_dict(torch.load('resnet50.pth'))

# 将模型设置为评估模式
model.eval()

# 定义预测函数
def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform['val'](image).unsqueeze(0).to(DEVICE)  # 对图像进行预处理，并添加一个维度
    with torch.no_grad():  # 关闭梯度计算
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # 获取预测结果中概率最大的类别
        return predicted.item()  # 返回预测结果的类别索引

# 进行预测
test_image_path = 'test_image.jpg'  # 替换为你的测试图像路径
predicted_class = predict_image(test_image_path)
if predicted_class == 0:
    print("预测结果：猫")
else:
    print("预测结果：狗")





