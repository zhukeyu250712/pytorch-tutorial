import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Hyper-parameters 
input_size = 28 * 28    # 784
num_classes = 10  # 分类的类别数（0-9）
num_epochs = 5  # 训练的轮数
batch_size = 100 # 每个小批量的样本数
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader (input pipeline) 创建数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Logistic regression model
model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()     # 交叉熵损失函数，用于多分类任务。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降（SGD）优化器，将模型的参数传递给优化器进行优化。

# Train the model
total_step = len(train_loader)
print("total_step:",total_step)

for epoch in range(num_epochs):
    # 使用enumerate函数获取每个小批量数据的索引 i 以及对应的 图像数据images 和 标签数据labels。
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        # 将其展平为大小为(batch_size, input_size)的张量。-1表示自动计算展平后的维度大小，input_size为每个图像的总像素数。
        images = images.reshape(-1, input_size)
        
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
# In test phase, we don't need to compute gradients (for memory efficiency)
# 用于测试训练好的逻辑回归模型在测试集上的准确率。
with torch.no_grad(): # 一个上下文管理器，用于在测试阶段关闭梯度计算，以提高内存效率。在该上下文中，不会计算梯度。
    correct = 0 # 正确预测的数量和总样本数量。
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)  # 将其展平为大小为(batch_size, input_size)的张量，与训练阶段保持一致。
        outputs = model(images)
        # 在每个输出张量中找到最大值及其对应的索引。_是用于忽略最大值，而predicted保存了预测的类别标签。
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)  # labels.size(0)返回当前小批量的样本数量
        correct += (predicted == labels).sum() # 计算预测与真实标签相等的数量。

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
