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
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage




S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'tsai-model-s1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'srgan.pt'
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


def fetch_input_image(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    print('Loading body...')
    print(event['body'])
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final picture that will be used by the model
    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    print('Picture obtained')

    image = Image.open(io.BytesIO(picture.content))
    return Variable(ToTensor()(image)).unsqueeze(0)


def srgan(event, context):
    """Super Resolution"""
    try:
        # Get image from the request
        image = fetch_input_image(event)

        
        out= model(image)
        output = ToPILImage()(out[0].data.cpu())

        # Convert output to bytes
        buffer = io.BytesIO()
        output.save(buffer, format="JPEG")
        output_bytes = base64.b64encode(buffer.getvalue())

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'data': output_bytes.decode('ascii'), 'size' : str(output.size)})
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
