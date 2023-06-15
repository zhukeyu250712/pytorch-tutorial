import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,  # 指定加载的是训练集数据。
                                           transform=transforms.ToTensor(),  # transforms.ToTensor()将图像数据转换为torch.Tensor类型，并将像素值缩放到范围[0, 1]之间。
                                           download=True)  # 指定是否下载数据集

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # 第一个卷积层，它包含了一个卷积操作(nn.Conv2d)、批归一化操作(nn.BatchNorm2d)、激活函数(nn.ReLU)和最大池化操作(nn.MaxPool2d)。该层输入通道数为1，输出通道数为16。
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # 卷积核大小为5x5，步长为1，填充为2。这个卷积操作对输入特征图进行滑动窗口卷积计算，生成16个输出特征图。
            nn.BatchNorm2d(16), # 对每个输出通道进行归一化处理，使得特征的分布更加稳定，有利于模型的训练和收敛。
            nn.ReLU(), # 对每个特征图的每个元素进行非线性映射，增加模型的非线性能力。
            nn.MaxPool2d(kernel_size=2, stride=2))  # 它对输入特征图进行2x2的窗口滑动，每次选取窗口中的最大值作为输出，以减小特征图的尺寸并提取最显著的特征。

        # 第二个卷积层的定义，它与第一层类似，但输入通道数为16，输出通道数为32。
        # 这些操作按照顺序被组合在nn.Sequential中，形成一个层级结构。这样，输入数据在前向传播时会依次经过卷积、批归一化、激活和池化操作，最终生成16个输出特征图。
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # 这是全连接层的定义，它接收展平后的特征图作为输入，输出大小为num_classes，即类别数。
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # 将最后一层的输出展平为一维向量，用于输入全连接层。
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device) # 将输入图像数据移动到所选的设备（GPU或CPU）上，以便进行计算。
        labels = labels.to(device) # 将标签数据移动到所选的设备上。
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# 将模型设置为评估模式。
# 在评估模式下，批归一化层（BatchNorm）会使用移动平均值和方差来计算归一化，而不是使用当前批次的均值和方差。这确保了在测试过程中的一致性，与训练过程中的归一化行为保持一致。
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
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

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')