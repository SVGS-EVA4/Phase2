try:
    import unzip_requirements
except ImportError:
    pass

import io
import json
import random
import numpy as np
import base64
import boto3
import torch
from PIL import Image
import os
from requests_toolbelt.multipart import decoder


S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'tsai-model-s1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'gan.pth'
print('Downloading model...')

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH)!=True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print('Creating Bytestream',obj)
        bytestream = io.BytesIO(obj['Body'].read())
        print('Loading model',bytestream)
        model = torch.jit.load(bytestream)
        print('Model loaded')

except Exception as e:
    print(repr(e))
    raise(e)

def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)
    
def generate(event, context):
    try:
        # Generate Image
       
        fixed_noise = torch.randn(64, 100, 1, 1, device='cpu')
        with torch.no_grad():
            fake = model(fixed_noise).detach().cpu()
        fake = denormalize(fake, (0.5,0.5,0.5), (0.5, 0.5, 0.5))
        generated_image = random.choice(fake).permute(1, 2, 0).numpy().copy() * 255
        generated_image = Image.fromarray(generated_image.astype(np.uint8))

        print('Loading output to buffer')
        buffer = io.BytesIO()
        generated_image.save(buffer, format="JPEG")
        generated_image_bytes = base64.b64encode(buffer.getvalue())

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'data': generated_image_bytes.decode('ascii')})
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
            'body': json.dumps({'error': repr(e)})
        }