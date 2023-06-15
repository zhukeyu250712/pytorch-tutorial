# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 80
batch_size = 100
learning_rate = 0.001

# Image preprocessing modules
# 使用了transforms.Compose来将这些图像预处理操作组合在一起,可以在训练过程中对输入图像进行数据增强和标准化处理，以提高模型的性能和泛化能力。
transform = transforms.Compose([
    transforms.Pad(4),   # 在图像的四周填充4个像素。这是为了扩展图像的尺寸，以便进行后续的随机裁剪操作。
    transforms.RandomHorizontalFlip(),  # 随机对图像进行水平翻转。这是一种数据增强技术，可以增加数据的多样性和模型的鲁棒性。
    transforms.RandomCrop(32),  # 随机裁剪图像为大小为32x32的区域。这也是一种数据增强技术，可以引入不同的视角和位置的图像样本。
    transforms.ToTensor()])  # 将图像转换为张量形式。这将把图像的像素值转换为范围在[0, 1]之间的浮点数，并将通道顺序从HWC（高度、宽度、通道）转换为CHW（通道、高度、宽度）。

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,  # 指定是否加载训练集。设置为True表示加载训练集。
                                             transform=transform, # 用于对图像进行预处理的转换操作。在这里，我们使用了之前定义的transform对象，它包含了一系列的图像预处理操作。
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 3x3 convolution
# 创建3x3卷积层的函数conv3x3。该函数接受输入通道数in_channels、输出通道数out_channels和步长stride作为参数，并返回一个nn.Conv2d实例。
# 在函数内部，使用nn.Conv2d来创建一个卷积层对象。它接受以下参数：
# in_channels: 输入图像的通道数。
# out_channels: 输出图像的通道数，也就是卷积层的滤波器数量。
# kernel_size: 卷积核的大小，这里设置为3x3。
# stride: 卷积核的步长，表示滑动窗口在输入图像上移动的距离，默认为1。
# padding: 填充的大小，用于保持输入和输出的尺寸一致，默认为1，表示在输入的边缘填充1个像素。
# bias: 是否使用偏置项，默认为False。
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
# 定义一个残差块（Residual Block）的类ResidualBlock，用于构建ResNet模型中的残差部分。
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()    # 调用父类的__init__方法进行初始化
        self.conv1 = conv3x3(in_channels, out_channels, stride)  # 第一个3x3卷积层，输入通道数为in_channels，输出通道数为out_channels，步长为stride。
        self.bn1 = nn.BatchNorm2d(out_channels)   # 第一个批归一化层，对out_channels个通道进行归一化。
        self.relu = nn.ReLU(inplace=True)   #表示使用ReLU激活函数，并将其应用在原地（inplace）。
        # 原地操作意味着直接在输入张量上进行修改，而不是创建一个新的张量。通过设置inplace=True，可以节省内存空间，因为不需要额外的内存来存储输出张量。
        self.conv2 = conv3x3(out_channels, out_channels)  # 第二个3x3卷积层，输入通道数为out_channels，输出通道数为out_channels。
        self.bn2 = nn.BatchNorm2d(out_channels)  # 第二个批归一化层，对out_channels个通道进行归一化。
        self.downsample = downsample   #如果存在下采样（stride > 1），则downsample是一个卷积层和批归一化层的组合，用于将输入x进行下采样以匹配残差路径上的维度。

    # 将输入x保存到residual中。然后，对输入x进行一系列的卷积、批归一化和ReLU操作，并将结果与residual相加。
    # 如果存在下采样，将输入x经过下采样操作后赋值给residual。最后，再次应用ReLU激活函数并返回输出。
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample: # 如果存在下采样操作（self.downsample不为None），则对输入x应用下采样操作（通过调用self.downsample(x)），以便将其尺寸调整为与残差块输出相匹配。
            residual = self.downsample(x)
        out += residual  # 将残差和输出相加（out += residual），以实现残差连接。最后，再次应用ReLU激活函数，并将输出返回。
        out = self.relu(out)
        return out

# ResNet
# 包含多个残差块的深度神经网络模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16           # 定义初始的输入通道数为16
        self.conv = conv3x3(3, 16)      # 创建一个3x3的卷积层,输入通道数为3（RGB图像）输出通道数为16。这个卷积层用于处理输入数据。
        self.bn = nn.BatchNorm2d(16)    # 创建一个批标准化层，对卷积层的输出进行标准化操作。
        self.relu = nn.ReLU(inplace=True)   # 创建一个ReLU激活函数层，并设置inplace=True表示原地操作，即将激活函数应用在输入上并覆盖原始输入。
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)  # 输入通道数为16，输出通道数为32，残差块数量为layers[1]，设置步长为2，用于下采样。
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)     # 平均池化层，用于对特征图进行降维操作。
        self.fc = nn.Linear(64, num_classes)  # 创建一个线性全连接层，将降维后的特征映射到类别数量的输出。

    # block是残差块的类型，out_channels是残差块的输出通道数，blocks是残差块的数量，stride是步长，默认值为1。
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):  # 步长不为1或输入通道数与输出通道数不相等，则需要进行下采样
            downsample = nn.Sequential(         # 卷积层和批标准化层的序列，用于实现下采样操作
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []     # 存储残差块
        layers.append(block(self.in_channels, out_channels, stride, downsample))  # 添加第一个残差块
        self.in_channels = out_channels    # 确保下一个残差块的输入通道数正确。
        for i in range(1, blocks):    # 从1到blocks-1，添加剩余的残差块到layers列表中，每个残差块的输入通道数和输出通道数都是out_channels。
            layers.append(block(out_channels, out_channels))

        # 用于按顺序组合多个神经网络层或模块。
        # 在这种情况下，*layers表示将layers列表中的每个元素作为参数传递给nn.Sequential。这样做的效果是将layers列表中的层按照顺序连接起来，形成一个层的序列。
        return nn.Sequential(*layers)   # 将layers列表转换为一个nn.Sequential对象，并返回该对象。
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)  # 将池化后的特征展平成一维向量，即将其形状变为(batch_size, -1)。
        out = self.fc(out)
        return out
    
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
# 更新优化器的学习率
# 这个函数通常在训练过程中使用，以动态调整学习率。在训练过程中，可以根据需要调用update_lr函数，以便在不同的训练阶段或条件下更新学习率，以提高训练效果。
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:  # 将每个参数组的学习率（lr）更新为指定的新值。
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
# 将模型设置为评估模式。
# 在评估模式下，批归一化层（BatchNorm）会使用移动平均值和方差来计算归一化，而不是使用当前批次的均值和方差。这确保了在测试过程中的一致性，与训练过程中的归一化行为保持一致。
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # sum()计算布尔张量中为True的元素个数，即预测正确的样本数。
        # .item()将结果从张量中提取为Python的标量值。
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')
