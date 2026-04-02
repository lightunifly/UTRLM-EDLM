import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
import multimolecule  # you must import multimolecule to register models
from multimolecule import RnaTokenizer, UtrLmModel

import torch
backbone = UtrLmModel.from_pretrained("multimolecule/utrlm-te_el")

x = torch.ones(1,1024,dtype=int)*4
with torch.cuda.amp.autocast(dtype=torch.float32):
    output = backbone(x)
    print('x:',x)
    print('output:',output)
with torch.cuda.amp.autocast(dtype=torch.float32):
    output = backbone(x)
    print('x:',x)
    print('output:',output)

output = backbone(x)
print('x:',x)
print('output:',output)

x = x.to('cuda')
print('x:',x)
backbone = backbone.to('cuda')
print('backbone:',backbone)
output = backbone(x)
print('x:',x)
print('output:',output)
from transformers import pipeline

print('a')
predictor = pipeline("fill-mask", model="multimolecule/utrlm-te_el")
output = predictor("gguc<mask>cucugguuagaccagaucugagccu")
print(output)

tokenizer = RnaTokenizer.from_pretrained("multimolecule/utrlm-te_el")
model = UtrLmModel.from_pretrained("multimolecule/utrlm-te_el")

text = "UAGCUUAUCAGACUGAUGUUG"
print('text:',text)
input = tokenizer(text, return_tensors="pt")
print('input:',input)
output = model(**input)
print('output:',output)

