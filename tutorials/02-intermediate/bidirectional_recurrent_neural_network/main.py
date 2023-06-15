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
learning_rate = 0.003

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
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

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    # input_size：输入特征的维度。
    # hidden_size：隐藏层的大小，也就是 LSTM 单元的数量。
    # num_layers：LSTM 层的数量。
    # num_classes：输出的类别数量。
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 创建一个双向的 LSTM 层。input_size 是输入特征的维度，hidden_size 是隐藏层的大小，num_layers 是 LSTM 层的数量
        # batch_first=True 表示输入的形状为 (batch_size, sequence_length, input_size)
        # 双向 LSTM 会同时在正向和反向两个方向上处理输入序列，并将它们的输出进行拼接
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # 创建一个线性层，将双向 LSTM 的输出特征转换为预测的类别数量。hidden_size*2 是因为双向 LSTM 的输出特征是正向和反向输出拼接在一起的，所以乘以 2
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        # 初始的隐藏状态和细胞状态为全零张量。由于是双向 LSTM，所以隐藏状态和细胞状态的数量是 self.num_layers*2
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        # 将输入数据 x 输入到双向 LSTM 中，获取输出 out。out 的形状是 (batch_size, sequence_length, hidden_size2)，其中 hidden_size2 是因为双向 LSTM 输出特征的拼接
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        # many-to-one 只对最后一个输出处理
        # 最后一个时间步的输出特征提取出来，并输入到线性层 self.fc 中，得到最终的预测结果
        out = self.fc(out[:, -1, :])
        return out

model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
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
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')