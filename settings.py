import json
SEED = 112
buckets = [(10, 10), (15, 15), (25, 25), (50, 50)]
split_ratio = 0.9
with open('replace_words.json','r') as f:
    replace_words = json.load(f)
