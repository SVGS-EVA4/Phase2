import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import math
import time
from pynput.keyboard import Key, Controller

keyboard = Controller()

class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]

model = keras.models.load_model('handrecognition_model.h5')

cap = cv2.VideoCapture(0)
paused = False

while(1):
        
    try:
        time.sleep(0.2)
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        
        #define region of interest
        roi=frame[100:300, 100:300]
        
        
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # Converts into the correct colorspace (GRAY)
        img = cv2.resize(img, (320, 120))
		
        X = np.array(img, dtype="uint8")
        X = X.reshape(1, 120, 320, 1)
        
        prediction = model.predict(X) # Make predictions
        predicted_label = np.argmax(prediction)
        print(predicted_label)
		
        title='frame'
        cv2.namedWindow(title)
        cv2.moveWindow(title, 0, 0)
        cv2.imshow(title, img)
        if(predicted_label == 1):
          if not paused:
            keyboard.press(Key.space)
            keyboard.release(Key.space)
            paused=True
        else:
            if paused:
              keyboard.press(Key.space)
              keyboard.release(Key.space)
              paused=False
    
    except:
        pass
        
    
    k = cv2.waitKey(5)
    if k == ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()    