import pandas as pd
from functools import reduce
import argparse 
import os
dirname = os.path.dirname(os.path.abspath(__file__))
default_dir = os.path.join(dirname,'xhj_dropout/xhj_300_4_32_jieba_s')
default_dir = os.path.join(dirname,'xhj/xhj_300_4_32_jieba_s')

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default=default_dir, help="prefix")
args = parser.parse_args()

prefix = args.prefix
df = pd.read_csv("%s_%s.csv"%(prefix,"change"))
df.columns = ["id","source","target","change"]
df_join = df[["id","change"]]
df_left = df[["source","target"]]

files = [0.2,0.5,0.75,1]
dfs = []
for f in files:
  if len(f) > 0: f = "_%s"%f 
  df = pd.read_csv("%s%s.csv"%(prefix,f))
  df.columns = ["id","source","target",f]
  df = df[["id",f]]
  dfs.append(df)
dfs.append(df_join)
df = reduce(lambda x,y: x.merge(y,on="id") ,dfs)

df = pd.concat([df[["id"]],df_left,df.drop("id",axis=1)],axis=1)
df.to_csv("%s_combined.csv"%prefix,index=False)
