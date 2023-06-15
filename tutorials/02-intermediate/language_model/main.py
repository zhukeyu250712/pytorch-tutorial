# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model 
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
embed_size = 128        # 嵌入层的大小
hidden_size = 1024      # 隐藏层的大小
num_layers = 1          # LSTM 层的数量
num_epochs = 5          # 训练的轮数
num_samples = 1000      # number of words to be sampled 生成文本时的样本数
batch_size = 20         # 每个批次的大小
seq_length = 30         # 序列长度
learning_rate = 0.002

# Load "Penn Treebank" dataset
corpus = Corpus()
ids = corpus.get_data('../../data/requirements.txt', batch_size)
vocab_size = len(corpus.dictionary)         # 词汇表的大小
num_batches = ids.size(1) // seq_length     # 训练过程中将数据分成多少个批次


# RNN based language model
class RNNLM(nn.Module):
    # vocab_size：词汇表的大小，表示模型可以预测的不同词的数量
    # embed_size：词嵌入的维度，表示将每个词映射到的连续向量空间的维度
    # hidden_size：隐藏状态的维度，表示RNN模型中隐藏状态的大小
    # num_layers：RNN的层数，表示RNN模型中堆叠的RNN层的数量
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        # 词嵌入层，用于将词索引映射为词嵌入向量。它的输入大小为vocab_size，输出大小为embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM层，它接收词嵌入向量作为输入，并通过多个LSTM层的堆叠来学习序列的上下文信息。它的输入大小为embed_size，输出大小为hidden_size。
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # 线性层，用于将LSTM层的输出映射回词汇表的大小，以进行词的预测。它的输入大小为hidden_size，输出大小为vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)

    # x是输入的词索引序列，形状为(batch_size, sequence_length)，其中batch_size表示批次大小，sequence_length表示序列长度。
    # h是隐藏状态，包含了LSTM的隐藏状态和细胞状态，形状为(num_layers, batch_size, hidden_size)。
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)       # 得到词嵌入向量x
        
        # Forward propagate LSTM
        # 生成LSTM的输出out和更新后的隐藏状态h和细胞状态c。
        out, (h, c) = self.lstm(x, h)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        # 将LSTM的输出out进行形状重塑，变为(batch_size*sequence_length, hidden_size)的形状
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        # 将重塑后的输出out通过线性层self.linear进行映射，得到预测的词的分布
        out = self.linear(out)
        return out, (h, c)      # 返回输出out和更新后的隐藏状态(h, c)

model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Truncated backpropagation
# 将LSTM的隐藏状态和细胞状态从计算图中分离，以实现截断反向传播
# 将分离后的状态作为下一个时间步的输入隐藏状态。这样可以有效地控制梯度的传播，并防止梯度在时间上的累积
# states是一个包含隐藏状态和细胞状态的列表或元组
def detach(states):
    return [state.detach() for state in states] 

# Train the model
for epoch in range(num_epochs):
    # Set initial hidden and cell states
    # 在每个epoch开始时，初始化隐藏状态h和细胞状态c为零张量
    # num_layers表示LSTM层数，batch_size表示每个mini-batch的样本数量，hidden_size表示隐藏状态的维度
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    
    for i in range(0, ids.size(1) - seq_length, seq_length):    # 生成每个mini-batch的起始位置，确保每个mini-batch之间有重叠
        # Get mini-batch inputs and targets
        inputs = ids[:, i:i+seq_length].to(device)              # 获取输入序列
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)     # 获取目标序列
        
        # Forward pass
        states = detach(states)     # 将上一个时间步的隐藏状态和细胞状态分离出来，避免梯度回传到过去的时间步
        outputs, states = model(inputs, states)     # 将输入序列和分离后的状态传递给模型，执行前向传播，获取模型的输出序列和更新后的隐藏状态和细胞状态
        loss = criterion(outputs, targets.reshape(-1))      # 计算损失，将输出序列和目标序列进行比较
        
        # Backward and optimize
        optimizer.zero_grad()       # 优化器的梯度置零
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)        # 对模型的参数进行梯度裁剪，以防止梯度爆炸
        optimizer.step()                                # 执行优化器的更新步骤

        step = (i+1) // seq_length              # 计算当前步数，用于判断是否需要打印训练状态
        if step % 100 == 0:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                   .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))   # 打印当前epoch、当前步数、损失值以及困惑度（Perplexity）

# Test the model
# 生成文本样本
with torch.no_grad():
    with open('sample.txt', 'w') as f:
        # Set intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # Select one word id randomly
        prob = torch.ones(vocab_size)
        # 随机选择一个单词作为起始输入
        # torch.multinomial(prob, num_samples=1) 根据概率分布prob进行多项式抽样，返回抽样结果的索引
        # unsqueeze(1)将抽样结果的维度从(1, ) 扩展为(1, 1)，以与模型的输入要求一致。
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            # Forward propagate RNN 
            output, state = model(input, state)

            # Sample a word id
            prob = output.exp()     # 输出的对数概率转换为概率
            word_id = torch.multinomial(prob, num_samples=1).item()     # 根据概率分布进行多项式抽样，并将抽样结果转换为单个整数值

            # Fill input with sampled word id for the next time step
            # 将input张量填充为抽样得到的word_id，以便在下一个时间步中作为模型的输入
            input.fill_(word_id)

            # File write
            # 将抽样得到的单词通过corpus.dictionary.idx2word[word_id]转换为实际的单词
            word = corpus.dictionary.idx2word[word_id]
            # 如果抽样得到的单词是特殊标记<eos>，则将word设置为换行符\n，否则将word设置为实际单词加上一个空格
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))

# Save the model checkpoints
torch.save(model.state_dict(), 'model.ckpt')