try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import boto3
import base64
from requests_toolbelt.multipart import decoder
import torch
import pickle


VOCAB_PATH = 'vocab.pickle'
vocab_file = open(VOCAB_PATH, 'rb')      
vocabs = pickle.load(vocab_file) 

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'eva4p2-s1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'TracedCSAmodel.pt'
print('Downloading model...')

s3 = boto3.client('s3')
try:
    if os.path.isfile(MODEL_PATH)!=True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print('Creating Bytestream')
        bytestream = io.BytesIO(obj['Body'].read())
        print('Loading model')
        model = torch.jit.load(bytestream)
        print('Model loaded')
except Exception as e:
    print(repr(e))
    raise(e)
		


	
def predict_sentiment(model, sentence, min_len = 5):
	model.eval()

	tokenized = sentence.split()
	if len(tokenized) < min_len:
		tokenized += ['<pad>'] * (min_len - len(tokenized))
	indexed = [vocabs.stoi[t] for t in tokenized]
	tensor = torch.LongTensor(indexed)

	tensor = tensor.unsqueeze(0)
	prediction = (torch.sigmoid(model(tensor))).item()
	print('prediction',prediction)
	return prediction

def get_sentiment(event,context):
	try:
		print('event',event)
		sentence = json.loads(event['body'])['data']
		print('sentence',sentence,type(sentence))
		sentence = str(sentence)
		prediction = predict_sentiment(model,sentence)
		predict = str(prediction)
		
		return {
				'statusCode': 200,
				'headers': {
					'Content-Type': 'application/json',
					# 'Access-Control-Allow-Origin': '*',
					# 'Access-Control-Allow-Credentials': True
				},
				'body': json.dumps({'Status':'1','data':predict})
			}
	except Exception as e:
		print(repr(e))
		return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                # 'Access-Control-Allow-Origin': '*',
                # 'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'Status':'0','error': repr(e)})
        }
