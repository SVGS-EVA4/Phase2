try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import boto3
import base64

import torch
import numpy as np
import pickle
from model import *
from utils import *

print("Imports Complete")

ger_vocab_path = 'german_vocab.pickle'
eng_vocab_path = 'english_vocab.pickle'
model_path = 'gru_model_cpu.pt'



gerfile = open(ger_vocab_path, 'rb')      
ger_vocab = pickle.load(gerfile) 


engfile = open(eng_vocab_path, 'rb')    
eng_vocab = pickle.load(engfile) 



model = make_model(len(ger_vocab), len(eng_vocab),
                   emb_size=256, hidden_size=256,
                   num_layers=1, dropout=0.2)


checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.eval()

print('Loaded model and vocabs')


def translator_german_english(event,context):
  try:
    print('event',event)

    sentence = json.loads(event['body'])['data']
    print('sentence',sentence,type(sentence))

    sentence = str(sentence)

    word_embedding = words_to_embedding(sentence,ger_vocab)
    print("word embedding",word_embedding)

    src = np.array([word_embedding])
    src_mask = np.array([[[ True for _ in range(len(src[0]))]]])
    src_lengths = np.array([len(src[0])])

    src = torch.Tensor(src).to(torch.int64)
    src_mask = torch.BoolTensor(src_mask)
    src_lengths = torch.Tensor(src_lengths).to(torch.int64)
    
    print('src:  ',src,'src_mask: ',src_mask,'src_lengths: ',src_lengths)
    print('src:  ',src.size(),'src_mask: ', src_mask.size(),'src_lengths: ', src_lengths.size())

    res = german_to_eng(src,src_mask,src_lengths, model, n=2, max_len=100, 
                      sos_index=1, 
                      src_eos_index=None,  
                      src_vocab=ger_vocab, trg_vocab=eng_vocab)
    print("res",res)

    result_str = str(res).translate(str.maketrans("","","[,']"))
    print("result_str",result_str)

    return {
				'statusCode': 200,
				'headers': {
					'Content-Type': 'application/json',
					'Access-Control-Allow-Origin': '*',
					'Access-Control-Allow-Credentials': True
				},
				'body': json.dumps({'Status':'1','data':result_str})
			}
  except Exception as e:
    print(repr(e))
    return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'Status':'0','error': repr(e)})
        }
