import pandas as pd
from functools import reduce
import argparse 
import os
dirname = os.path.dirname(os.path.abspath(__file__))
default_dir = os.path.join(dirname,'ptt_dropout/ptt_300_4_32_jieba_s')
default_dir = os.path.join(dirname,'ptt/ptt_300_4_32_jieba_s')

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default=default_dir, help="prefix")
parser.add_argument("--epoch", default='', help="epoch")
args = parser.parse_args()

prefix = args.prefix
df = pd.read_csv("%s_%s.csv"%(prefix,"change"))
df.columns = ["id","source","target","change"]
df_join = df[["id","change"]]
df_left = df[["source","target"]]

files = ['',0.2,0.5,0.75,1]
dfs = []
for f in files:
  if len(str(f)) > 0: f = "_%s"%f 
  df = pd.read_csv("%s%s.csv"%(prefix,f))
  if f == "": f = "seq2seq"
  df.columns = ["id","source","target",f]
  df = df[["id",f]]
  dfs.append(df)
dfs.append(df_join)
df = reduce(lambda x,y: x.merge(y,on="id") ,dfs)

df = pd.concat([df[["id"]],df_left,df.drop("id",axis=1)],axis=1)
if len(args.epoch) > 0:
    csv_name = "%s_combined_%s.csv"%(prefix,args.epoch)
else:
    csv_name = "%s_combined.csv"%prefix
df.to_csv(csv_name,index=False)
