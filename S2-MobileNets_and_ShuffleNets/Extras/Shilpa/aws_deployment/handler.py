try:
    import unzip_requirements
except ImportError:
    pass
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import boto3
import os
import io
import json
import base64
from requests_toolbelt.multipart import decoder
print('Import End...')

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'eva4p2-s1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'e4p2_s2_model_best.pt'

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

def toSquare_img(img_bytes):
    print('padding the image')
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
  
    h,w = img.size[0],img.size[1]
    max_len = max(h,w)
    if h == w:
        return img
        
    elif h>w:
        diff = int(abs(h-w)/2)
        black = np.zeros((max_len,max_len))
        black_img = Image.fromarray(black,mode='RGB')

        black_img.paste(img,(0,diff))
        return black_img
    elif w>h:
        diff = int(abs(h-w)/2)
        black = np.zeros((max_len,max_len))
        black_img = Image.fromarray(black,mode='RGB')

        black_img.paste(img,(diff,0))

        return black_img


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.531,0.586,0.615],std=[0.282,0.257,0.294]),
            
            
        ])
        image = toSquare_img(image_bytes)
        print('Successfully padded the image')
        # image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

def get_prediction(image_bytes):
    print('applying augmentation')
    tensor = transform_image(image_bytes = image_bytes)
    return model(tensor).argmax().item()

def classify_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event['body'])
        print('body loaded')

        picture = decoder.MultipartDecoder(body,content_type_header).parts[0]
        prediction = get_prediction(image_bytes= picture.content)
        print('model predictions completed')
        classes = ['Winged_Drones', 'Small_QuadCopters', 'Large_QuadCopters', 'Flying_Birds' ]
        predicted_class = classes[int(prediction)]
        print(prediction,predicted_class)

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
        
        return {
            'statusCode': 200,
            'headers':{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'File': filename.replace('"',''), 'Predicted Class': predicted_class })
        }
    except Exception as e:
        print(repr(e))
        return {
            'statusCode': 500,
            'headers':{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({ 'error': repr(e) })
        }


