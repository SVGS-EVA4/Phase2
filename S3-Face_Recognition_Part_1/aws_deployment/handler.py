try:
    
    import unzip_requirements
    print('imported unzip requirements')
except ImportError:
    pass

import json
import base64
from requests_toolbelt.multipart import decoder
from PIL import Image
print('importing ..')
import cv2
print('Imported opencv')
import dlib
print('Imported dlib')
import numpy as np
import math
import io
print('Import COmplete')

predictor_path = 'shape_predictor_5_face_landmarks.dat'
print('loaded landmarks file')



# detect facial landmarks in image
def getLandmarks(faceDetector, landmarkDetector, im, FACE_DOWNSAMPLE_RATIO = 1):
    points = []
    imSmall = cv2.resize(im,None,
                       fx=1.0/FACE_DOWNSAMPLE_RATIO, 
                       fy=1.0/FACE_DOWNSAMPLE_RATIO, 
                       interpolation = cv2.INTER_LINEAR)
  
    faceRects = faceDetector(imSmall, 0)
  
    if len(faceRects) > 0:
        maxArea = 0
        maxRect = None
        # TODO: test on images with multiple faces
        for face in faceRects:
            if face.area() > maxArea:
                maxArea = face.area()
                maxRect = [face.left(),
                        face.top(),
                        face.right(),
                        face.bottom()
                        ]
    
        rect = dlib.rectangle(*maxRect)
        scaledRect = dlib.rectangle(int(rect.left()*FACE_DOWNSAMPLE_RATIO),
                                int(rect.top()*FACE_DOWNSAMPLE_RATIO),
                                int(rect.right()*FACE_DOWNSAMPLE_RATIO),
                                int(rect.bottom()*FACE_DOWNSAMPLE_RATIO))
        
        landmarks = landmarkDetector(im, scaledRect)
        points = dlibLandmarksToPoints(landmarks)
    return points

# convert Dlib shape detector object to list of tuples
def dlibLandmarksToPoints(shape):
    points = []
    for p in shape.parts():
        pt = (p.x, p.y)
        points.append(pt)
    return points

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)

    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    # The third point is calculated so that the three points make an equilateral triangle
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

    inPts.append([np.int(xin), np.int(yin)])

    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

    outPts.append([np.int(xout), np.int(yout)])

    # Now we can use estimateAffine2D for calculating the similarity transform.
    tform = cv2.estimateAffine2D(np.array([inPts]), np.array([outPts]), False)
    return tform

def normalizeImagesAndLandmarks(outSize,imIn,pointsIn):

    h, w = outSize

    # Corners of the eye in the input image
    if len(pointsIn) == 68:
        eyecornerSrc = [pointsIn[36],pointsIn[45]]
    elif len(pointsIn) == 5:
        eyecornerSrc = [pointsIn[2],pointsIn[0]]

    # Corners of the eye i  normalized image
    eyecornerDst = [(np.int(0.3*w),np.int(h/3)),(np.int(0.7*w),np.int(h/3))]

    # Calculate similarity transform
    tform = similarityTransform(eyecornerSrc, eyecornerDst)

    imOut = np.zeros(imIn.shape, dtype=imIn.dtype)

    # Apply similarity transform to input image
    imOut = cv2.warpAffine(imIn, tform[0], (w,h))

    # reshape pointsIn from numLandmarks x 2 to  numLandmarks x 1 x 2
    points2 = np.reshape(pointsIn,(pointsIn.shape[0],1,pointsIn.shape[1]))

    # Apply similarity transform to landmarks
    pointsOut = cv2.transform(points2,tform[0])

    # reshape pointsOut to numLandmarks x 2
    pointsOut = np.reshape(pointsOut,(pointsIn.shape[0],pointsIn.shape[1]))

    return imOut, pointsOut


def align_face(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event['body'])
        print('body loaded')
        print(content_type_header)
        picture = decoder.MultipartDecoder(body,content_type_header).parts[0]

        ####################################################################################################
        print(picture)
        im_arr = np.frombuffer(picture.content, dtype=np.uint8)
        print('picture2')
        im = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        print('img decoded')

        faceDetector = dlib.get_frontal_face_detector()
        landmarkDetector = dlib.shape_predictor(predictor_path)
        print('initialized detectors')
        if len(faceDetector(im,0)) == 0:
            return {
            'statusCode': 200,
            'headers':{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({ 'Status':'IncorrectInput','Result':'No faces detected','ImageBytes': ''  })
            }
        else:
            points = getLandmarks(faceDetector,landmarkDetector,im)
            print('landmarks= ',points)
            print(len(points))
            points = np.array(points)

            im = np.float32(im)/255.0

            h = 500
            w = 500
            print('Aligning face ...')
            imNorm, points = normalizeImagesAndLandmarks((h,w),im,points)
            print('face aligned')
            imNorm = np.uint8(imNorm*255)
            
            img = Image.fromarray(imNorm[:,:,::-1])
            print('converting to bytes')
            byte_arr = io.BytesIO()
            print('encoding image bytes to base64')
            img.save(byte_arr, format='JPEG')
            encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii') 
            print(encoded_img)
            
            return {
                'statusCode': 200,
                'headers':{
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Credentials': True
                },
                'body': json.dumps({ 'Status':'Success','Result':'Faces detected','ImageBytes': encoded_img  })
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
