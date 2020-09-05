try:
    
    import unzip_requirements
    print('imported unzip requirements')
except ImportError:
    pass

import json
import cv2
from operator import itemgetter
import albumentations as A
from PIL import Image
import base64
from requests_toolbelt.multipart import decoder
import os
import io
import onnxruntime
import numpy as np

print('import complete')


MODEL_PATH = 'human_pose_model.onnx'

print('Downloading model...')

try:
        ort_session = onnxruntime.InferenceSession(MODEL_PATH)
        print('Model loaded')
except Exception as e:
    print(repr(e))
    raise(e)

def aug_list(p=1):
    return A.Compose([
          A.Resize(256,256),

          A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                  ],p=p)

def transform_image(image_bytes):
    try:        
        image = Image.open(io.BytesIO(image_bytes))
        im = np.asarray(image)
        augmentation = aug_list()
        data = {"image": im }
        augmented = augmentation(**data)
        image = augmented["image"]
        return image
    except Exception as e:
        print(repr(e))
        raise(e)




get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])


def human_pose_estimation(event,context):

    try:
        content_type_header = event['headers']['content-type']
        print('body')
        body = base64.b64decode(event['body'])
        print('body loaded')
        picture = decoder.MultipartDecoder(body,content_type_header).parts[0]

        print('picture')

		# key-points connection
        POSE_PAIRS = [
        # UPPER BODY
                    [9, 8],[8, 7],[7, 6],
        # LOWER BODY
                    [6, 2],[2, 1],[1, 0],[6, 3],[3, 4],[4, 5],
        # ARMS
                    [7, 12],[12, 11],[11, 10],[7, 13],[13, 14],[14, 15]
        ]

        JOINTS = ['r-ankle', 'r-knee', 'r-hip', 'l-hip', 'l-knee', 'l-ankle', 'pelvis', 'thorax', 'upper-neck', 'head-top', 'r-wrist', 'r-elbow', 'r-shoulder', 'l-shoulder', 'l-elbow', 'l-wrist']
        THRESHOLD = 0.5

	# apply image augmentation

        tensor = transform_image(image_bytes = picture.content)	
        print('aug applied')	
        tensor1 = np.transpose(tensor,(2,0,1))
        tensor1 = tensor1[np.newaxis,... ]
        print('tensor1.shape',tensor1.shape)
        ort_inputs = {ort_session.get_inputs()[0].name: tensor1}
        ort_outs = ort_session.run(None, ort_inputs)
        img_pose = np.array(ort_outs[0][0])
        print(img_pose.shape)
		
        res_height = img_pose.shape[-1]
        res_width = img_pose.shape[-2]

        OUT_SHAPE = (res_width,res_height)

        im_arr = np.frombuffer(picture.content, dtype=np.uint8)
        print('picture2')
        image_p = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        print('img decoded')

        pose_layers = img_pose		
        key_points = list(get_keypoints(pose_layers=pose_layers))

        print(key_points)
        print(len(key_points))
		
        is_joint_plotted = [False for i in range(len(JOINTS))]
        for pose_pair in POSE_PAIRS:
            from_j, to_j = pose_pair

            from_thr, (from_x_j, from_y_j) = key_points[from_j]
            to_thr, (to_x_j, to_y_j) = key_points[to_j]

            IMG_HEIGHT, IMG_WIDTH, _ = image_p.shape

            from_x_j, to_x_j = from_x_j * IMG_WIDTH / OUT_SHAPE[0], to_x_j * IMG_WIDTH / OUT_SHAPE[0]
            from_y_j, to_y_j = from_y_j * IMG_HEIGHT / OUT_SHAPE[1], to_y_j * IMG_HEIGHT / OUT_SHAPE[1]

            from_x_j, to_x_j = int(from_x_j), int(to_x_j)
            from_y_j, to_y_j = int(from_y_j), int(to_y_j)

            if from_thr > THRESHOLD and not is_joint_plotted[from_j]:
                # this is a joint
                cv2.ellipse(image_p, (from_x_j, from_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
                is_joint_plotted[from_j] = True

            if to_thr > THRESHOLD and not is_joint_plotted[to_j]:
                # this is a joint
                cv2.ellipse(image_p, (to_x_j, to_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
                is_joint_plotted[to_j] = True

            if from_thr > THRESHOLD and to_thr > THRESHOLD:
                # this is a joint connection, plot a line
                cv2.line(image_p, (from_x_j, from_y_j), (to_x_j, to_y_j), (255, 74, 0), 3)

        res = Image.fromarray(cv2.cvtColor(image_p, cv2.COLOR_RGB2BGR))
        byte_arr = io.BytesIO()
        res.save(byte_arr, format='JPEG')
        encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii') 
        print('complete')
        return {
            'statusCode': 200,
            'headers':{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({ 'Status':'Success','Result':'Pose estimation successful','ImageBytes': encoded_img  })
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
            'body': json.dumps({ 'Status':'Error','Result':'Error','error': repr(e) })
        }
