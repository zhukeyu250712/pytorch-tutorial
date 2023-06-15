import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Hyper-parameters
# 设置超参数
input_size = 1 # 输入的特征数量
output_size = 1 # 输出的特征数量
num_epochs = 60  # 训练轮数
learning_rate = 0.001   # 学习率

# Toy dataset
# 定义数据集
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Linear regression model
# 定义线性回归模型
model = nn.Linear(input_size, output_size)

# Loss and optimizer
# 定义损失函数和优化器
criterion = nn.MSELoss()    # 使用均方误差损失函数
# 使用随机梯度下降优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
# 训练模型
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    # 将numpy数组转换为torch张量
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
# detach().numpy(): 这一部分用于分离计算图并将输出转换回NumPy数组。
# detach()方法将输出张量与计算图分离，这样就不会影响后续的反向传播过程。然后，.numpy()方法将分离后的张量转换回NumPy数组形式。
predicted = model(torch.from_numpy(x_train)).detach().numpy()  # 将输入数据传递给模型model进行前向传播计算。
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
# 保存模型参数
# model.state_dict(): 这部分代码返回了神经网络模型的当前状态字典。状态字典是一个Python字典对象，它包含了模型的所有参数（权重和偏置等）以及对应的键值对。
torch.save(model.state_dict(), 'model.ckpt')