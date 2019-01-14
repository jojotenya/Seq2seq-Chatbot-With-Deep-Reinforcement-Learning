import numpy as np
import keras
import argparse
import os
dirname = os.path.dirname(os.path.abspath(__file__))

_dict = {
  "ptt":
    {"SPLIT_DIMS":[6,6,8],
     "model_name":"Model_ptt_xhj"
    },
  "sentiment":
    {"SPLIT_DIMS":[5,5,3],
     "model_name":"Model07"
    }
}

train_type = os.environ["style_type"] 

GO_ID = 0
EOS_ID = 1
UNK_ID = 2
SPLIT_DIMS = _dict[train_type]["SPLIT_DIMS"]
MAX_LEN = SPLIT_DIMS[0]*SPLIT_DIMS[1]*SPLIT_DIMS[2]
batch_size = 128

mapping_path = os.path.join(dirname,"char_mapping")
model_dir = os.path.join(dirname+"/saved_model/",_dict[train_type]["model_name"])
cut_mode = "char"

with open(mapping_path,"r") as f:
    vocab_dict = dict([(row.strip(),i) for i,row in enumerate(f.readlines())])

def text_to_sequence(row,mode="char"):
    if mode=="char":
        row = [r for r in row]
        row = list(map(lambda x:vocab_dict.get(x,UNK_ID),row))
        row = [GO_ID] + row
        row.append(EOS_ID)
    return row

def texts_to_sequences(rows,mode="char"):
    return list(map(lambda row:text_to_sequence(row,mode),rows))

def pad_sequence(tokens,maxlen):
    pad_num = maxlen - len(tokens)
    tokens += [EOS_ID]*pad_num
    return tokens

def pad_sequences(tokens_list,maxlen):
    return np.array(list(map(lambda tokens:pad_sequence(tokens,maxlen),tokens_list)))

def get_split_list(arr,dims): 
    arrs = []
    for i in range(arr.shape[0]):
        split1=np.split(arr[i],dims[2])
        a=[]
        for j in range(dims[2]):
            s=np.split(split1[j],dims[1])
            a.append(s)
        arrs.append(a)
    return arrs

def cal_score(s):
    idx = np.argmax(s)
    return idx*s[idx]

def cal_scores(scores):
    return list(map(lambda x:cal_score(x),scores))
