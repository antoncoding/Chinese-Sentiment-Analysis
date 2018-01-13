
# coding: utf-8

# In[2]:


from keras.models import model_from_json
from gensim.models import word2vec
import jieba
import pandas as pd
import numpy as np
import nltk
import pickle
import os

#from gensim.models import word2vec
from keras.engine.topology import Layer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss,f1_score
from keras.layers.recurrent import LSTM, GRU
from keras import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras import backend as K
from keras import utils
from keras import initializers, constraints, regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
K.set_learning_phase(1)

import pandas as pd
import logging

# ## Data Preprocessing

# In[3]:


jieba.set_dictionary("dict.txt")


# In[4]:


X_train_df = pd.read_csv("google_play.csv", delimiter="|", error_bad_lines=False).dropna(axis=0,subset=["comments"])


# In[5]:


Y_train = list(X_train_df["star"])


# In[6]:


X_train = list(X_train_df["comments"])


# In[7]:


print(X_train[1])


# In[9]:


X_train = [ i.replace("，","").replace(" ","") for i in X_train ]


# In[8]:


X_test = ["還蠻好用的",
          "常常會閃退....",
          "有些小缺點，但整體來說還算好用",
          "很不好用，不推薦大家使用",
          "沒用過這麼卡的軟體....",
          "介面很友善",
          "事實擺在眼前，何必狡辯！無恥之徒鬼話廢話謊話連篇",
          "打掃比打人簡單，試著學學看吧，加油!!",
          "可憐的基層公務人員，一天最熱的12:00-15:00被關冷氣然後說不缺電的智障總統爽爽吹提出節能關冷氣的高官爽爽吹",
          "了不起，負責!",
          "這關我屁事阿笑死",
          "這牌子的啤酒很好喝!",
          "這種藝人還是不要出來秀下限比較好ㄏㄏ",
          "配備拔掉這麼多，還賣這盤子價，台奧不意外",
          "油耗表現不好，不過內裝設計還不錯，勉強可以考慮",
          "技嘉主機板出名的爛你還敢買ㄏㄏ",
          "特斯拉好潮R，想買",
          "超愛玖壹壹!!",
          "太帥啦QQ",
          ]


# In[10]:


X_test_raw = X_test


# In[11]:


Y_train = np.asarray(Y_train,dtype=float)
Y_train_cat = utils.to_categorical(Y_train-1)


# In[12]:


X_TRAIN_LEN = len(X_train)


# In[13]:


All = X_train + X_test


# In[14]:


All_cut = []
for i in range(len(All)):
    All_cut.append([k + " " for k in jieba.cut(All[i],cut_all=False)])


# In[15]:


for i in range(len(All_cut)):
    flatten = ""
    for k in All_cut[i]:
        flatten += k
    All_cut[i] = flatten


# In[20]:


# out = open("model/rnn.txt","w")
# for i in All_cut:
#     out.write(i+"\n")


# In[16]:


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence('model/rnn.txt')


# In[17]:


DIM = 200
# model = word2vec.Word2Vec(sentences, size=DIM, iter=200, min_count=1, sample=0.000, workers=16)


# In[18]:


# model.save('word2vec_201801')


# In[20]:


model = word2vec.Word2Vec.load('word2vec_201801')


# In[21]:


All_cut = []
for i in range(len(All)):
    All_cut.append([k + " " for k in jieba.cut(All[i],cut_all=False) if k in model.wv.vocab])


# In[23]:


for i in range(len(All_cut)):
    flatten = ""
    for k in All_cut[i]:
        flatten += k
    All_cut[i] = flatten


# In[24]:


# filters = "(,\n].;)”’“&'" + '"' + "'"
# tokenizer = Tokenizer(num_words=100000,filters=filters)
# tokenizer.fit_on_texts(All_cut)
#
#
# # In[25]:
#
#
# with open('train_2.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('train_2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# In[26]:


vectors = []
vocabs = []
for items in model.wv.vocab:
    vectors.append(model.wv[items])
    vocabs.append(items.lower())
word_dic = {}
for vocab, vec in zip(vocabs,vectors):
    word_dic[vocab] = vec


# In[27]:


word_index = tokenizer.word_index
vocab_dim = DIM # dimensionality of your word vectors
n_symbols = len(word_index) + 1 # adding 1 to account for 0th index (for masking)
embedding_weights = np.zeros((n_symbols+1,vocab_dim))
count = 0
for word,index in word_index.items():
    if word in word_dic:
        embedding_weights[index,:] = word_dic[word]
        count += 1


# In[28]:


All_tokenized = tokenizer.texts_to_sequences(All_cut)
All_padded = pad_sequences(All_tokenized,maxlen=80)
word_index = tokenizer.word_index


# In[29]:


X_train = All_padded[:X_TRAIN_LEN]
X_test = All_padded[X_TRAIN_LEN:]


# In[30]:


embedding_dimension = DIM
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
for word, i in word_index.items():
    embedding_vector = word_dic.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector[:embedding_dimension]


# In[31]:


embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=X_train.shape[1], trainable=False)


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


# In[51]:


RNN_model = Sequential()
RNN_model.add(embedding_layer)
RNN_model.add(GRU(64, dropout = 0.3, recurrent_dropout = 0.3, return_sequences=False, implementation=2))
RNN_model.add(Dense(256, activation='relu'))
RNN_model.add(Dropout(0.45))
RNN_model.add(Dense(1, activation='relu'))


# In[52]:


opt = optimizers.rmsprop(lr=0.001,clipvalue=1)

RNN_model.compile(loss='mse', optimizer=opt)

print(RNN_model.summary())
RNN_model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs = 100, batch_size = 1500 , verbose=1)
#loaded_model.load_weights("0.526_scithresh0.285.h5")


# In[74]:


for(text, score) in zip(X_test_raw, RNN_model.predict(X_test)):
    print(text)
    print(str(score[0]) + '\n')


# # Model 2: Use Attention Layer

# In[65]:


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


## V1
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        a = K.exp(ait)


        if mask is not None:
            a *= K.cast(mask, K.floatx())


        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (input_shape[0], input_shape[-1])


# In[62]:


model_2 = Sequential()
model_2.add(embedding_layer)
model_2.add(Bidirectional(GRU(64, dropout = 0.3, recurrent_dropout = 0.3, return_sequences=True, implementation=2)))
model_2.add(AttentionWithContext())
model_2.add(Dense(256, activation='relu'))
model_2.add(Dropout(0.45))
model_2.add(Dense(1, activation='relu'))


# In[63]:


opt = optimizers.rmsprop(lr=0.001,clipvalue=1)
#model_2.compile(loss='mse', optimizer=opt)
#print(model_2.summary())
#model_2.fit(x_train, y_train, validation_data=(x_test,y_test), epochs = 100, batch_size = 1500 , verbose=1)
#loaded_model.load_weights("0.526_scithresh0.285.h5")


# In[64]:


#model_2.save("VER_2.0_1122.h5")


# In[73]:


# for(text, score) in zip(X_test_raw, model_2.predict(X_test)):
#     print(text)
#     print(str(score[0]) + '\n')


# # Model 3 : Using New Activation Function: Swish

# In[77]:


from keras import backend as K
from keras.layers import Activation

from keras.utils.generic_utils import get_custom_objects

def swish(x):
    return x * K.sigmoid(x)

get_custom_objects().update({'swish': Activation(swish)})


# In[78]:


model_3 = Sequential()
model_3.add(embedding_layer)
model_3.add(Bidirectional(GRU(64, dropout = 0.3, recurrent_dropout = 0.3, return_sequences=True, implementation=2)))
model_3.add(AttentionWithContext())
model_3.add(Dense(256, activation='swish'))
model_3.add(Dropout(0.45))
model_3.add(Dense(1, activation='swish'))


# In[ ]:


opt = optimizers.rmsprop(lr=0.001,clipvalue=1)
model_3.compile(loss='mse', optimizer=opt)
# print(model_3.summary())
# model_3.fit(x_train, y_train, validation_data=(x_test,y_test), epochs = 100, batch_size = 1500 , verbose=1)
#
#
# # In[ ]:
#
#
# model_3.save("VER_3.0_1122.h5")
