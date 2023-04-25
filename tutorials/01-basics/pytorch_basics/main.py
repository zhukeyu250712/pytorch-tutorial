import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189) 


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #
print("1. Basic autograd example 1 ")
# Create tensors.
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3
print("y: ", y)

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 


# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #
print("\n2. Basic autograd example 2")
# Create tensors of shape (10, 3) and (10, 2).
# input
x = torch.randn(10, 3)
# 真实值
y = torch.randn(10, 2)

# Build a fully connected layer.
linear = nn.Linear(3, 2)

# weight表示该层的权重
# weight是一个二维tensor,它的维度决定了该层的输入和输出大小。具体来说:
# - weight的shape是(out_features, in_features)
# - in_features表示该层的输入特征数,即输入的第二个维度
# - out_features表示该层的输出特征数,即输出的第二个维度
print ('w: ', linear.weight)
# bias表示该层的偏置
# bias是一个一维tensor,它的维度只与该层的输出特征数相关
# - bias的长度是out_features
# - 它为输出的各个维度增加一个偏差量
print ('b: ', linear.bias)
print("----------parameters-----------")
for param in linear.parameters() :
    print(param)

# Build loss function and optimizer.
# 定义均方误差损失函数(Mean Squared Error loss)。
criterion = nn.MSELoss()

# SGD的主要思想是:
# 1. 随机选取一个样本(或批量样本),根据网络对该样本的预测计算出误差/损失。
# 2. 计算损失函数关于权重参数的梯度。
# 3. 根据梯度更新权重参数,通常是参数 = 参数 - 学习率 * 梯度。
# 4. 重复步骤1~3,不断更新权重从而使损失下降。
# 这样通过不断使用随机的样本更新权重,网络可以朝着使整体损失下降的方向学习。
# 创建一个SGD优化器用于更新网络权重,学习率为0.01
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass.
# 得到预测值
pred = linear(x)

# Compute loss.
# 计算均方误差损失（预测值pred，真实值y）
loss = criterion(pred, y)

# 优化器梯度清零
# optimizer.zero_grad()

print('loss: ', loss.item())

# Backward pass.
# 反向传播
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
# 权重更新
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #
print("\n3. Loading data from numpy")

# Create a numpy array.
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)
print("y: ",y)

# Convert the torch tensor to a numpy array.
z = y.numpy()
print("z: ",z)


# ================================================================== #
#                         4. Input pipeline                           #
# ================================================================== #
print("\n4. Input pipeline")

# Download and construct CIFAR-10 dataset.
# CIFAR-10是一个常用的图片分类数据集,它包含60000张32x32的彩色图片,分为10个类别
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

# Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print (image.size())
print (label)

# Data loader (this provides queues and threads in a very simple way).
# torch.utils.data.DataLoader是PyTorch中对数据的封装,用于方便地iterate数据集。
# 它做了以下工作:
# 1. 将数据集分成批量(batch),使网络能够逐批训练。
# 2. 如果数据集过大难以放入内存,可以通过DataLoader实现数据的流式加载( streaming)。
# 3. 可以自动打乱(shuffle)和采样(sampling)数据集。
# 4. 在多GPU或多机环境下可以通过num_workers参数实现数据的并行加载。
# 5. 数据的预处理操作(transform)可以统一在DataLoader中完成,简化代码。
# 使用DataLoader的常见步骤是:
# 1. 定义dataset,可以是TensorDataset、ImageFolder等
# 2. 定义数据预处理操作transform(可选)
# 3. 创建DataLoader
# 4. iterate数据集并训练
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass

# ================================================================== #
#                5. Input pipeline for custom dataset                 #
# ================================================================== #
print("\n5. Input pipeline for custom dataset")

# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 

# You can then use the prebuilt data loader. 
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=True)


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #
print("6. Pretrained model")

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #
print("7. Save and load the model")

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
