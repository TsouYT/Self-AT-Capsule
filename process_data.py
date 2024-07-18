import numpy as np
import h5py
import re
import operator
from collections import defaultdict
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

cachedStopWords = stopwords.words("english")

def load_word_vector(fname, vocab):
    model = {}
    with open(fname) as fin:
        for line_no, line in enumerate(fin):
            try:
                parts = line.strip().split(' ')
                word, weights = parts[0], parts[1:]
                if word in vocab:                     
                    model[word] = np.array(weights,dtype=np.float32)
            except:
                pass
    return model

def load_bin_vec(model, vocab):
    word_vecs = {}
    for word in vocab:
        if word in model:
            word_vecs[word] = model[word]
    return word_vecs

def line_to_words(line):
    words = map(lambda word: word.lower(), word_tokenize(line))
    tokens = words
    p = re.compile('[a-zA-Z]+')
    return list(filter(lambda token: p.match(token) and len(token) >= 3, tokens))        

def get_vocab(dataset):
    max_sent_len = 0
    word_to_idx = {}
    idx = 1
    for line in dataset:    
        words = line_to_words(line)
        max_sent_len = max(max_sent_len, len(words))
        for word in words:
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    return max_sent_len, word_to_idx

def load_txt(path_name):
    f = open(path_name, 'r')
    sentences = f.readlines()
    f.close()
    trains, trains_label = [], []
    for sentence in sentences:
        if sentence is None or len(sentence) <= 1:
            continue
        train = sentence[2:-2]
        train_label = sentence[:1]
        trains.append(train)   
        trains_label.append(train_label)  

    train_docs, test_docs, train_cats, test_cats = train_test_split(trains, trains_label, test_size=0.1, random_state=0)
    return train_docs, train_cats, test_docs, test_cats

def load_data(path_name, padding=0, sent_len=65, w2i=None):
    threshold = 1 
    
    train_docs, train_cats, test_docs, test_cats = load_txt(path_name)
    dataset = train_docs + test_docs
    max_sent_len, word_to_idx = get_vocab(dataset)

    if sent_len > 0:
        max_sent_len = sent_len      
    if w2i is not None:
        word_to_idx = w2i    

    train, train_label, test, test_label = [], [], [], []
    
    for i, line in enumerate(train_docs):
        words = line_to_words(line)
        y = train_cats[i]
        sent = [word_to_idx[word] for word in words if word in word_to_idx]
        if len(sent) > max_sent_len:
            sent = sent[:max_sent_len]
        else:    
            sent.extend([0] * (max_sent_len + padding - len(sent)))
        train.append(sent)
        train_label.append(y)
    
    single_label = ['-1'] + list(set(train_label))
    num_classes = len(single_label)

    for i, l in enumerate(train_label):
        train_label[i] = single_label.index(l)

    for i, line in enumerate(test_docs):
        words = line_to_words(line)
        y = test_cats[i]    
        sent = [word_to_idx[word] for word in words if word in word_to_idx]
        if len(sent) > max_sent_len:
            sent = sent[:max_sent_len]
        else:    
            sent.extend([0] * (max_sent_len + padding - len(sent)))
        test.append(sent)    
        one_hot_y = np.zeros([num_classes], dtype=np.int32)
        for yi in y:
            one_hot_y[single_label.index(yi)] = 1
        test_label.append(one_hot_y)

    return single_label, word_to_idx, np.array(train), np.array(train_label), np.array(test), np.array(test_label)

path_name = 'mpqa.all'
single_label, word_to_idx, train, train_label, test, test_label = load_data(path_name, padding=0, sent_len=39, w2i=None)

print(train[0])
print(single_label)
print('train size:', train.shape)
print('test size:', test.shape)
print('train_label size:', train_label.shape)
print('test_label size:', test_label.shape)

dataset = 'mpqa'
with open(dataset + '_word_mapping.txt', 'w+') as embeddings_f:
    embeddings_f.write("*PADDING* 0\n")
    for word, idx in sorted(word_to_idx.items(), key=operator.itemgetter(1)):
        embeddings_f.write("%s %d\n" % (word, idx))

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  

w2v = load_bin_vec(model, word_to_idx)
print('Word embeddings loaded.')

V = len(word_to_idx) + 1
print('Vocab size:', V) 

def compute_embed(V, w2v):
    np.random.seed(1)
    embed = np.random.uniform(-0.25, 0.25, (V, next(iter(w2v.values())).shape[0]))
    for word, vec in w2v.items():
        embed[word_to_idx[word]] = vec
    return embed

embed_w2v = compute_embed(V, w2v)

filename = dataset + '.hdf5'
with h5py.File(filename, "w") as f:
    f["w2v"] = np.array(embed_w2v)
    f['train'] = train
    f['train_label'] = train_label
    f['test'] = test
    f['test_label'] = test_label

print('Data saved to', filename)

####################################################################################################
# import numpy as np
# import h5py
# import re
# import operator

# from collections import defaultdict
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split

# # 加載停用詞
# cachedStopWords = stopwords.words("english")

# def line_to_words(line):
#     """
#     將句子分割成單詞，並且只保留長度大於等於3的字母單詞
#     """
#     words = map(lambda word: word.lower(), word_tokenize(line))
#     p = re.compile('[a-zA-Z]+')
#     return list(filter(lambda token: p.match(token) and len(token) >= 3, words))

# def get_vocab(dataset):
#     """
#     生成詞彙表並計算句子最長長度
#     """
#     max_sent_len = 0
#     word_to_idx = {}
#     idx = 1
#     for line in dataset:
#         words = line_to_words(line)
#         max_sent_len = max(max_sent_len, len(words))
#         for word in words:
#             if word not in word_to_idx:
#                 word_to_idx[word] = idx
#                 idx += 1
#     return max_sent_len, word_to_idx

# def load_txt(path_name):
#     """
#     加載文本文件並將句子和標籤分開
#     """
#     with open(path_name, 'r', encoding='utf-8') as f:
#         sentences = f.readlines()
#     trains, trains_label = [], []
#     for sentence in sentences:
#         if sentence is None or len(sentence) <= 1:
#             continue
#         train = sentence[2:-2]
#         train_label = sentence[:1]
#         trains.append(train)
#         trains_label.append(train_label)
#     return trains, trains_label

# def load_data(file_train, file_dev, file_test, padding=0, sent_len=65, w2i=None):
#     """
#     加載數據並進行預處理，返回詞彙表和訓練、驗證、測試數據
#     """
#     train_docs, train_cats = load_txt(file_train)
#     dev_docs, dev_cats = load_txt(file_dev)
#     test_docs, test_cats = load_txt(file_test)
    
#     # 合併訓練和驗證數據
#     combined_docs = train_docs + dev_docs
#     combined_cats = train_cats + dev_cats

#     all_labels = list(set(combined_cats + test_cats))
#     num_classes = len(all_labels)
    
#     dataset = combined_docs + test_docs
#     max_sent_len, word_to_idx = get_vocab(dataset)

#     if sent_len > 0:
#         max_sent_len = sent_len      
#     if w2i is not None:
#         word_to_idx = w2i    

#     train, train_label = [], []
#     for i, line in enumerate(combined_docs):
#         words = line_to_words(line)
#         y = combined_cats[i]
#         sent = [word_to_idx[word] for word in words if word in word_to_idx]
#         if len(sent) > max_sent_len:
#             sent = sent[:max_sent_len]
#         else:
#             sent.extend([0] * (max_sent_len + padding - len(sent)))
#         train.append(sent)
#         train_label.append(all_labels.index(y))
    
#     test, test_label = [], []
#     for i, line in enumerate(test_docs):
#         words = line_to_words(line)
#         y = test_cats[i]
#         sent = [word_to_idx[word] for word in words if word in word_to_idx]
#         if len(sent) > max_sent_len:
#             sent = sent[:max_sent_len]
#         else:
#             sent.extend([0] * (max_sent_len + padding - len(sent)))
#         test.append(sent)
#         one_hot_y = np.zeros([num_classes], dtype=np.int32)
#         one_hot_y[all_labels.index(y)] = 1
#         test_label.append(one_hot_y)
    
#     return word_to_idx, np.array(train), np.array(train_label), np.array(test), np.array(test_label), num_classes

# # 指定文件路徑
# file_train = 'stsa.binary.phrases.train'
# file_dev = 'stsa.binary.dev'
# file_test = 'stsa.binary.test'

# # 加載數據
# word_to_idx, train, train_label, test, test_label, num_classes = load_data(file_train, file_dev, file_test, padding=0, sent_len=39, w2i=None)

# print('數據加載成功!')
# print(f'類別數量: {num_classes}')
# print(f'訓練集大小: {train.shape}')
# print(f'測試集大小: {test.shape}')

# dataset = 'sst2'
# with open(dataset + '_word_mapping.txt', 'w+', encoding='utf-8') as embeddings_f:
#     embeddings_f.write("*PADDING* 0\n")
#     for word, idx in sorted(word_to_idx.items(), key=operator.itemgetter(1)):
#         embeddings_f.write("%s %d\n" % (word, idx))

# import gensim
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  

# print('Word embeddings loaded.')

# V = len(word_to_idx) + 1
# print('Vocab size:', V)

# def compute_embed(V, w2v, word_to_idx):
#     np.random.seed(1)
#     embed = np.random.uniform(-0.25, 0.25, (V, w2v.vector_size))
#     for word, idx in word_to_idx.items():
#         if word in w2v:
#             embed[idx] = w2v[word]
#     return embed

# embed_w2v = compute_embed(V, model, word_to_idx)

# filename = dataset + '.hdf5'
# with h5py.File(filename, "w") as f:
#     f["w2v"] = np.array(embed_w2v)
#     f['train'] = train
#     f['train_label'] = train_label
#     f['test'] = test
#     f['test_label'] = test_label

# print('Data saved to', filename)

