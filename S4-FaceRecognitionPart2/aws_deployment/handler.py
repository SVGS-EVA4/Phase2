try:
    import unzip_requirements
except ImportError:
    pass
import dlib
import cv2
from PIL import Image
import numpy as np
import os
import io
import json
import base64
from requests_toolbelt.multipart import decoder
import onnxruntime
from helperFunctions import *
print('Import End...')

model_path = 'face_recognition_model.onnx'
face_predictor_path = 'shape_predictor_5_face_landmarks.dat'

def align_face(img_bytes,face_predictor_path):
    try:
        im_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        print('picture2')
        im = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        print('img decoded')
        size = im.shape
        faceDetector = dlib.get_frontal_face_detector()
        landmarkDetector = dlib.shape_predictor(face_predictor_path)
        print('initialized detectors')
        if len(faceDetector(im,0)) == 0:
            return {
            'statusCode': 200,
            'headers':{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({ 'Status':'0','Result':'No faces detected','ImageBytes': ''  })
            }
        else:
            points = getLandmarks(faceDetector,landmarkDetector,im)
            print('landmarks= ',points)
            print(len(points))
            points = np.array(points)

            im = np.float32(im)/255.0
            
            
            h = size[0]
            w = size[1]
            print('Aligning face ...')
            imNorm, points = normalizeImagesAndLandmarks((h,w),im,points)
            print('face aligned')
            imNorm = np.uint8(imNorm*255)
            
            img = Image.fromarray(imNorm[:,:,::-1])
            return img

    except Exception as e:
        print(repr(e))
        raise(e)




def aug_list():
    return A.Compose([
          A.Resize(224,224,p=1),
          A.HorizontalFlip(always_apply=False, p=0.5),
          A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],p=1),
                  ])

def transform_image(img):
    try:        
        
        im = np.asarray(img)
        augmentation = aug_list()
        data = {"image": im }
        augmented = augmentation(**data)
        image = augmented["image"]
        return image
    except Exception as e:
        print(repr(e))
        raise(e)

def get_prediction(image_array, model_path):
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: image_array}
    ort_outs = ort_session.run(None, ort_inputs)
    res = np.array(ort_outs[0][0])

    return res.argmax()


def recognise_face(event, context):
    try:
        content_type_header = event['headers']['content-type']
        # print(event['body'])
        body = base64.b64decode(event['body'])
        print('body loaded')

        picture = decoder.MultipartDecoder(body,content_type_header).parts[0]

        
        aligned_face = align_face(img_bytes = picture.content,face_predictor_path=face_predictor_path)

        nparray = transform_image(img = aligned_face)	
        print('aug applied')	
        nparray1 = np.transpose(nparray,(2,0,1))
        nparray1 = nparray1[np.newaxis,... ]
        print('nparray1.shape',nparray1.shape)
        


        prediction = get_prediction(image_array= nparray1, model_path = model_path)

        print('model predictions completed')
        classes = {0:'APJ Abdul Kalam', 15:'Barack Obama',26:'Chandler Bing',39:'Elon Musk',89:'Joey Tribianni',127:'Michelle Obama',160:'Ross Gellar',149:'Rachel Green',146:'Pheobe Buffay',131:'Monica Gellar'}
         
        if int(prediction) in classes:
            predicted_class = classes[int(prediction)]
        else:
            predicted_class = 'Unknown'
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
            'body': json.dumps({'Status':'1','File': filename.replace('"',''), 'Predicted_Class': str(predicted_class) ,'Class_No.':str(prediction) })
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
            'body': json.dumps({ 'error': repr(e) ,'Status':'0' })
        }

