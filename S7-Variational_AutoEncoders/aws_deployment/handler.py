try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import base64
import boto3
from PIL import Image
from requests_toolbelt.multipart import decoder
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'eva4p2-s1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'vae_traced_model.pt'

print('Downloading model...')

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH)!=True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print('Creating Bytestream')
        bytestream = io.BytesIO(obj['Body'].read())
        print('Loading model')
        vae_model = torch.jit.load(bytestream)
        print('Model loaded')
except Exception as e:
    print(repr(e))
    raise(e)

def fetch_input_image(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    print('Loading body...')
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final picture that will be used by the model
    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    print('Picture obtained')
    
    return picture.content


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize((128,128)),
  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

def reconstruct(event, context):
    """Reconstruct the Input Image."""
    try:
        # Get image from the request
        picture = fetch_input_image(event)

        print('applying aug to img')
        model_input = transform_image(picture)
        output = vae_model(model_input)

        output = output[0].squeeze(0)
        output1 = np.transpose(output.detach().numpy(), (1, 2, 0))
        output2 = Image.fromarray((output1 * 255).astype(np.uint8))

        # Convert output to bytes
        buffer = io.BytesIO()
        output2.save(buffer, format="JPEG")
        output_bytes = base64.encodebytes(buffer.getvalue())

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                # 'Access-Control-Allow-Origin': '*',
                # 'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'Status':'1','data': output_bytes.decode('ascii')})
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
            'body': json.dumps({'Status':'1','error': repr(e)})
        }
