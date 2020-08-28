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
from helperFunctions import *
print('Import COmplete')

predictor_path = 'shape_predictor_68_face_landmarks.dat'
print('loaded landmarks file')


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

def face_swap(event,context):

    try:
        # Read images

        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event['body'])
        print('body loaded')
        print(content_type_header)

        picture1 = decoder.MultipartDecoder(body,content_type_header).parts[0]
        print(picture1)
        im_arr1 = np.frombuffer(picture1.content, dtype=np.uint8)
        print('picture1')
        img1 = cv2.imdecode(im_arr1, flags=cv2.IMREAD_COLOR)
        print('img1 decoded')

        picture2 = decoder.MultipartDecoder(body,content_type_header).parts[0]
        print(picture2)
        im_arr2 = np.frombuffer(picture2.content, dtype=np.uint8)
        print('picture2')
        img2 = cv2.imdecode(im_arr2, flags=cv2.IMREAD_COLOR)
        print('img2 decoded')



        img1Warped = np.copy(img2)


        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        # Read array of corresponding points
        points1 = getLandmarks(detector, predictor, img1)
        points2 = getLandmarks(detector, predictor, img2)
        if len(detector(img1,0)) == 0 or len(detector(img2,0)) == 0:
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
            # Find convex hull
            hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
            print('convex hull generated')
            # Create convex hull lists
            hull1 = []
            hull2 = []
            for i in range(0, len(hullIndex)):
                hull1.append(points1[hullIndex[i][0]])
                hull2.append(points2[hullIndex[i][0]])

            # Calculate Mask for Seamless cloning
            hull8U = []
            for i in range(0, len(hull2)):
                hull8U.append((hull2[i][0], hull2[i][1]))

            mask = np.zeros(img2.shape, dtype=img2.dtype) 
            cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
            print('mask created')
            # Find Centroid
            m = cv2.moments(mask[:,:,1])
            center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))

            # Find Delaunay traingulation for convex hull points
            sizeImg2 = img2.shape    
            rect = (0, 0, sizeImg2[1], sizeImg2[0])

            dt = calculateDelaunayTriangles(rect, hull2)

            # If no Delaunay Triangles were found, quit
            if len(dt) == 0:
                quit()

            tris1 = []
            tris2 = []
            for i in range(0, len(dt)):
                tri1 = []
                tri2 = []
                for j in range(0, 3):
                    tri1.append(hull1[dt[i][j]])
                    tri2.append(hull2[dt[i][j]])

                tris1.append(tri1)
                tris2.append(tri2)

            # Simple Alpha Blending
            # Apply affine transformation to Delaunay triangles
            for i in range(0, len(tris1)):
                warpTriangle(img1, img1Warped, tris1[i], tris2[i])
                
            # Clone seamlessly.
            output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
            print('face swapped. Done!')
            res = Image.fromarray(output[:,:,::-1])
            print('converting to bytes')
            byte_arr = io.BytesIO()
            print('encoding image bytes to base64')
            res.save(byte_arr, format='JPEG')
            encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii') 
            print(encoded_img)
            
            return {
                'statusCode': 200,
                'headers':{
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Credentials': True
                },
                'body': json.dumps({ 'Status':'Success','Result':'Faces Swapped','ImageBytes': encoded_img  })
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
