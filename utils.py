import re
import jieba
from settings import buckets,split_ratio,SEED
import numpy as np

len_2 = ['樂']
def qulify_sentence(sentence):
    w = []
    tmp = []
    for i in range(len(sentence)):
        if i == 0:
            w.append(sentence[i])
            tmp.append(sentence[i])
        else:
            if sentence[i] in len_2:
                limit_num = 1
            else:
                limit_num = 2
            if len(tmp)>limit_num:
                if sentence[i] == tmp[-1]: continue
                else:
                    tmp = [sentence[i]]
                    w.append(sentence[i])
            else:
                if sentence[i] == tmp[-1]:
                    tmp.append(sentence[i])
                else:
                    tmp = [sentence[i]]
                w.append(sentence[i])
    return ('').join(w)

def sub_words(word):
    for rep in replace_words.keys():
        if rep in word:
            word = re.sub(rep,replace_words[rep],word)
    return word

def word_seg(input_file,output_file,mode):
    if mode == 'word':
        jieba.load_userdict('dict.txt')
    
    with open(output_file,'w') as f, open(input_file,'r') as fi:
        for l in fi:
            # remove all whitespace characters
            l = ''.join(l.split())
            if mode == 'char':
                f.write(' '.join(list(l)) + '\n')
            else:
                seg = jieba.cut(l, cut_all=False)
                f.write(' '.join(seg) + '\n')

def split_train_val(source,target,buckets=buckets):
    data = [[] for i in range(len(buckets))]
    with open(source,'r') as src, open(target,'r') as trg:
        src = list(src)
        np.random.seed(SEED)
        np.random.shuffle(src)
        src = iter(src)
        trg = list(trg)
        np.random.seed(SEED)
        np.random.shuffle(trg)
        trg = iter(trg)
        for s,t in zip(src,trg):
            sl, tl = len(s.split()), len(t.split())
            for bucket_id, (source_size, target_size) in enumerate(buckets):         
                if sl < source_size and tl < target_size:
                    data[bucket_id].append((s, t, sl, tl))
                    break

    with open(source+'_train', 'w') as src_train,\
         open(source+'_val', 'w') as src_val,\
         open(target+'_train', 'w') as trg_train,\
         open(target+'_val','w') as trg_val:

        for b, ds in zip(buckets, data):
            dl = len(ds)
            print('\n')
            print(b)
            print('data : ' + str(dl))
            for i, d in enumerate(ds):
                (s, t, sl, tl) = d
                if i < int(dl*split_ratio):
                    src_train.write(s)
                    trg_train.write(t)
                else:
                    src_val.write(s)
                    trg_val.write(t)
    
