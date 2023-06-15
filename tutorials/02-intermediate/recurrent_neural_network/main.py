import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28    # 输入序列的长度，这里表示序列中的时间步数
input_size = 28         # 每个时间步的输入元素的大小。在这里，它表示每个时间步的特征数
hidden_size = 128       # RNN 隐藏状态中的单元数
num_layers = 2          # RNN 模型中的循环层数量
num_classes = 10        # 分类任务中的输出类别数
batch_size = 100        # 训练过程中每个小批次的样本数
num_epochs = 2          # 在训练期间整个训练数据集上迭代的次数
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),  # 指定对数据进行的转换操作，这里使用 transforms.ToTensor() 将图像数据转换为张量格式。
                                           download=True)

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

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()     # 调用父类 nn.Module 的构造函数，确保模型被正确地初始化
        # 输入的隐藏状态大小和 RNN 层数保存为类的成员变量，以便在整个类中访问
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True 参数表示输入数据的维度顺序为 (batch_size, seq_length, input_size)
        # 其中 batch_size 是批量大小，seq_length 是序列长度，input_size 是输入的特征大小。这样设置后，我们可以直接将输入数据的形状定义为 (batch_size, seq_length, input_size)。
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)   # 将最后一个时间步的隐藏状态映射到输出类别的数量
    
    def forward(self, x):
        # Set initial hidden and cell states
        # 创建初始的隐藏状态和细胞状态张量，大小为 (num_layers, batch_size, hidden_size)。这些张量将用作 LSTM 层的初始隐藏状态和细胞状态。
        # self.num_layers 是 LSTM 层的层数，表示有多少个堆叠的 LSTM 层。
        # x.size(0) 表示输入数据的批量大小，即输入数据的第一个维度大小。
        # self.hidden_size 表示隐藏状态的维度大小。
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        #  LSTM 层处理输入数据 x 和初始隐藏状态 (h0, c0)，并返回输出张量 out。out 的形状为 (batch_size, seq_length, hidden_size)，表示每个时间步的隐藏状态。
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        # 提取最后一个时间步的隐藏状态，并通过线性层 self.fc 将其映射到输出类别的数量。最终的输出 out 的形状为 (batch_size, num_classes)。
        out = self.fc(out[:, -1, :])
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):                                 # 外层循环迭代每个训练周期（epoch）
    for i, (images, labels) in enumerate(train_loader):         # 内层循环迭代每个训练批次（batch)
        # 将输入图像张量 images 调整形状为 (batch_size, sequence_length, input_size)，以适应 RNN 模型的输入要求
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()   # 清除之前计算的梯度
        loss.backward()
        optimizer.step()

        # 打印当前训练批次的损失，其中包括当前训练周期的编号、当前批次的编号和损失值
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# 将模型设置为评估模式
# 在评估模式下，批归一化层（BatchNorm）会使用移动平均值和方差来计算归一化，而不是使用当前批次的均值和方差。这确保了在测试过程中的一致性，与训练过程中的归一化行为保持一致
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # 在数据张量的第一个维度上执行最大值操作，并返回最大值和对应的索引。第一个参数是数据张量，第二个参数 1 表示在每一行上执行最大值操作
        # 最大值操作返回两个值，用下划线 _ 表示我们不关心最大值本身，而 predicted 是包含每个样本预测的最大值索引的张量
        _, predicted = torch.max(outputs.data, 1)   # 从输出中获取预测类别，并获得每个样本预测的最大值及其对应的索引
        total += labels.size(0)     # 累加总样本数,当前批次的样本数量累加到总样本数 total 中
        correct += (predicted == labels).sum().item()   # 计预测正确的样本数

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')