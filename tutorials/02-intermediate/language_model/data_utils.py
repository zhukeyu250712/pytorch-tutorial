import torch
import os


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}      # 将单词映射到索引
        self.idx2word = {}      # 将索引映射回单词
        self.idx = 0            # 计数器，用于追踪词典中的单词数量

    # 用于向词典中添加新单词
    def add_word(self, word):
        # 如果单词不在 word2idx 中，将其添加到 word2idx 中，并更新相应的索引映射关系。同时，更新 idx 的值以反映词典中的新单词数量。
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)       # 返回词典中的单词数量，即 word2idx 的长度


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    # 从给定的文件路径中获取数据，并返回一个处理后的数据张量。
    # 该方法首先将文件中的单词添加到词典中，然后将文件内容进行分词处理，并将分词后的单词转换为对应的索引。
    # 最后，将索引数据转换为一个张量，并根据指定的批量大小进行重新排列。
    def get_data(self, path, batch_size=20):
        # Add words to the dictionary
        with open(path, 'r') as f:      # 打开指定路径的文件，并将其赋值给变量 f
            tokens = 0
            for line in f:      # 对于每一行，首先使用 split() 方法将行拆分成单词列表，然后将 <eos> 标记添加到列表末尾，以表示句子的结束
                words = line.split() + ['<eos>']
                tokens += len(words)    # 计算该行中的单词数，并将其累加到变量 tokens 中，以计算整个文件中的总单词数
                for word in words:      # 遍历该行中的每个单词，并使用 self.dictionary.add_word(word) 将单词添加到语料库的词典对象中。这样，词典会逐渐建立起单词到索引的映射关系。
                    self.dictionary.add_word(word)  
        
        # Tokenize the file content
        # 将文件内容进行标记化处理，将单词映射为对应的索引值
        ids = torch.LongTensor(tokens)      # 创建了一个大小为 tokens 的长整型张量（LongTensor） ids，这样做是为了在存储标记时使用整数索引而不是浮点数。用于存储标记化后的内容
        token = 0   # 索引
        with open(path, 'r') as f:
            for line in f:      # # 对于每一行，首先使用 split() 方法将行拆分成单词列表，然后将 <eos> 标记添加到列表末尾，以表示句子的结束
                words = line.split() + ['<eos>']
                for word in words:      # 遍历每个单词，通过 self.dictionary.word2idx[word] 将单词映射为对应的索引，并将索引值存储到 ids 张量中的相应位置
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size     # 表示可以形成多少个完整的 batch
        ids = ids[:num_batches*batch_size]          # ids 张量进行裁剪，使其大小符合 batch 大小要求，即去掉多余的部分
        return ids.view(batch_size, -1)             # 返回形状为 (batch_size, -1) 的 ids 张量，其中 -1 表示根据张量的大小自动计算对应的维度大小，以保证所有数据都能被正确分组成 batch