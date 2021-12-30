import tensorflow as tf
import numpy as np
from Bio import SeqIO
from propy.QuasiSequenceOrder import GetSequenceOrderCouplingNumberTotal
from sklearn import metrics
from propy.PseudoAAC import GetAAComposition
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

def BPF(seq):
    code = []
    tem =[]
    k = 7
    for i in range(k):
        if seq[i] =='A':
            tem = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='C':
            tem = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='D':
            tem = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='E':
            tem = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='F':
            tem = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='G':
            tem = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='H':
            tem = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='I':
            tem = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='K':
            tem = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='L':
            tem = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='M':
            tem = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='N':
            tem = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif seq[i]=='P':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif seq[i]=='Q':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif seq[i]=='R':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif seq[i]=='S':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif seq[i]=='T':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif seq[i]=='V':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif seq[i]=='W':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif seq[i]=='Y':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        code += tem
    return code
def AAC(seq):
    code= GetAAComposition(seq)
    res=[]
    for v in code.values():
        res.append(v)
    return res
def CKSAAGP(fastas, gap = 5):
    '''计算被任意k个残基隔开的氨基酸对的频率根据物理化学性质组成的不同组别'''

    def generateGroupPairs(groupKey):
        '''CKSAAGP的子函数'''
        gPair = {}
        for key1 in groupKey:
            for key2 in groupKey:
                gPair[key1 + '.' + key2] = 0
        return gPair
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1+'.'+key2)

    encodings = []

    name, sequence = [], fastas
    code = [name]
    for g in range(gap + 1):
        gPair = generateGroupPairs(groupKey)
        sum = 0
        for p1 in range(len(sequence)):
            p2 = p1 + g + 1
            if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[index[sequence[p1]] + '.' + index[
                    sequence[p2]]] + 1
                sum = sum + 1

        if sum == 0:
            for gp in gPairIndex:
                code.append(0)
        else:
            for gp in gPairIndex:
                code.append(gPair[gp] / sum)
    code = code[1:]
    return code
def DPC(seq: str):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    name, sequence = [], seq
    code = [name]
    tmpCode = [0] * 400
    for j in range(len(sequence) - 2 + 1):
        tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
            sequence[j + 1]]] + 1
    if sum(tmpCode) != 0:
        tmpCode = [i / sum(tmpCode) for i in tmpCode]
    code = code + tmpCode
    return code[1:]
def SOCNumber(seq:str):
    code=GetSequenceOrderCouplingNumberTotal(seq)
    res=[]
    for v in code.values():
        res.append(v)
    return res

def encode(seq):
    x1 = BPF(seq)
    x2 = AAC(seq)
    x3 = CKSAAGP(seq)
    x4 = DPC(seq)
    x5 = SOCNumber(seq)

    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    x4 = np.array(x4)
    x5 = np.array(x5)


    res = np.concatenate([x1,x2,x3,x4,x5], axis=-1)
    return res

def calculate_specificity(y_num, y_pred, y_true):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(y_num):
        if y_true[i] == 1:
            if y_true[i] == y_pred[i]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_true[i] == y_pred[i]:
                tn = tn + 1
            else:
                fp = fp + 1
    specificity = float(tn) / (tn + fp)
    return  specificity

batchsize=50
epochs=30

train = list(SeqIO.parse('E:/acp/ACPred-FL/train.txt', format="fasta"))
test = list(SeqIO.parse('E:/acp/ACPred-FL/test.txt', format="fasta"))

protein_dict = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,
                'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}

train_p_seq = []
train_n_seq = []
for x in train:
    id = str(x.id)
    if id.endswith("1"):
        seq = str(x.seq)
        train_n_seq.append(seq)
    if id.endswith("0"):
        seq = str(x.seq)
        train_p_seq.append(seq)

tr_seqs = train_n_seq + train_p_seq
tr_label = [0] * len(train_n_seq) + [1] * len(train_p_seq)

te_p_seq = []
te_n_seq = []
for x in test:
    id = str(x.id)
    if id.endswith("1"):
        seq = str(x.seq)
        te_n_seq.append(seq)
    if id.endswith("0"):
        seq = str(x.seq)
        te_p_seq.append(seq)

te_seqs = te_n_seq + te_p_seq
te_label = [0] * len(te_n_seq) + [1] * len(te_p_seq)


# 模型的第一个输入，进行embedding
# 数据集按照字典进行编码
train1 = []
for seq in tr_seqs:
    tmp = []
    for i in seq:
        tmp.append(protein_dict[i])
    train1.append(tmp)

test1=[]
for seq in te_seqs:
    tmp = []
    for i in seq:
        tmp.append(protein_dict[i])
    test1.append(tmp)

# 找出最长的序列长度
maxlen = 0
for i in train1:
    if len(i) >= maxlen:
        maxlen = len(i)
for i in test1:
    if len(i) >= maxlen:
        maxlen = len(i)

# 不够长的序列填0
train1 = pad_sequences(train1, maxlen=maxlen, padding='post')
test1 = pad_sequences(test1, maxlen=maxlen, padding='post')

# 模型的第二个输入，将数据集按照特征函数进行编码
train2 = []
for i in tr_seqs:
    train2.append(encode(i))

test2 = []
for i in te_seqs:
    test2.append(encode(i))

# train1是第一个通道，train2是第二个通道
# 第一个通道先embedding，再biLSTM
# 第二个通道是直接将数据集进行特征函数编码
train1 = np.array(train1)
train2 = np.array(train2)
train_label = np.array(tr_label)

test1 = np.array(test1)
test2 = np.array(test2)


# 模型
# 通道1是embedding biLSTM
# 通道2是特征编码
# 通道1 2进行concatenate
# 进入全连接层
fealen = train2.shape[-1]

in_x1 = tf.keras.layers.Input(shape=(maxlen,))
in_x2 = tf.keras.layers.Input(shape=fealen,)     #各个特征编码维度加起来的结果

x1 = tf.keras.layers.Embedding(21, 64)(in_x1)
x1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x1)

x2 = tf.keras.layers.Dense(512, 'relu')(in_x2)

x = tf.keras.layers.Concatenate(axis=-1)([x1, x2])
x = tf.keras.layers.Dense(512, 'relu')(x)

out_x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=[in_x1, in_x2], outputs=[out_x])
model.summary()

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate = 6e-3),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

# estop = tf.keras.callbacks.EarlyStopping(patience=13, monitor="val_loss", mode="min")
history = model.fit([train1, train2], train_label,
					batch_size=batchsize,
					epochs=epochs,
					shuffle=True,
                    # callbacks=[estop],
                    validation_split=0.2,
                    verbose=1)

# dic = history.history
# loss_ = dic['loss']
# acc_ = dic['binary_accuracy']
# epoch=range(1,len(loss_)+1)
# plt.plot(epoch,loss_,'r',label='loss')
# plt.plot(epoch,acc_,'b',label='acc')
# plt.title('the loss and acc of train')
# plt.xlabel('epochs')
# plt.legend()
# plt.show()

# 预测
pred_res = model.predict([test1,test2])
t = 0.5
pred_label = []
for x in pred_res:
    if x >= t:
        pred_label.append(1)
    else:
        pred_label.append(0)

acc = metrics.accuracy_score(y_pred=pred_label, y_true=te_label)
mcc = metrics.matthews_corrcoef(y_pred=pred_label, y_true=te_label)
# f1 = metrics.f1_score(y_pred=pred_label, y_true=te_label)
sensitivity = metrics.recall_score(y_pred=pred_label, y_true=te_label)
specificity = calculate_specificity(len(te_label), y_pred=pred_label, y_true=te_label)
auc = metrics.roc_auc_score(y_true=te_label, y_score=pred_res)
# precision = metrics.precision_score(y_pred=pred_label, y_true=te_label)


# print('pre:',precision)
print('se:',sensitivity)
print('sp:',specificity)
print('acc:',acc)
print('mcc:',mcc)
# print('f1:',f1)
print('auc:',auc)

fpr, tpr, thresholds = metrics.roc_curve(te_label, pred_res, pos_label=1)

plt.rc('font',family='Times New Roman')
plt.figure(dpi=600)
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc='lower right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

# precision_, recall_,thresholds_  = metrics.precision_recall_curve(te_label, pred_res, pos_label=1)
# plt.plot(recall_, precision_, 'b')
# plt.plot([0, 1], [1, 0], 'k--')
# # plt.legend(loc='lower right')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.1])
# plt.xlabel('recall')
# plt.ylabel('precision')
# plt.title('PR Curve')
# plt.show()
precision_, recall_,thresholds_  = metrics.precision_recall_curve(te_label, pred_res, pos_label=1)
plt.figure(dpi=600)
plt.plot(recall_, precision_, 'b')
plt.plot([0, 1], [1, 0], 'k--')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR Curve')
ax1 = plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.show()
