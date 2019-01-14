import argparse
import numpy as np
import os
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
parser = argparse.ArgumentParser()
parser.add_argument("--limit", "-l", default=200, dest="limit", help="index limit to")
parser.add_argument("--offset", "-o", default=10, dest="offset", help="offset how many index")
args = parser.parse_args()

model_dir = os.environ["model_dir"]
loss_val_file = os.path.join(model_dir,"loss_val")
loss_train_file = os.path.join(model_dir,"loss_train")
#1 unit = 500
limit_to = int(args.limit)
offset = int(args.offset) 

with open(loss_val_file, "r") as f:
    ave = 0
    vals = []
    for i,d in enumerate(f.readlines()):
        if i !=0 and (i+1)%5==0:
            vals.append(ave/4.0)
            ave=0
        else:
            ave+=float(d)

with open(loss_train_file, "r") as f:
    trains = [float(row.strip()) for row in f.readlines()] 

print('model_dir: ',model_dir)
df = pd.DataFrame({"train":trains,"val":vals}).iloc[offset:limit_to]
p = df.plot()
fig = p.get_figure()
fig.savefig('outputs/figures/%s.png'%(os.environ["dir_base"]))
