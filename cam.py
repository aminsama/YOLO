import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {#option is passes as dictionary
    'model': 'cfg/yolo.cfg',#loading config for model
    'load': 'bin/yolov2.weights',#loading weights from pretrained models 
    'threshold': 0.2, #parameter for getting box lower threshold= higher no of box
    'gpu': 1.0 #using gpu 
}

tfnet = TFNet(options)#passing option onto tfnet
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)#passing your web cam as input device
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)#setting res pix width of 1920
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)#setting res pix hight of 1080

while True:
    stime = time.time()#keep count of time for per frame
    ret, frame = capture.read()#reading from our capture device ie webcam
    if ret:#loop for our webcam if its rec as a stream
        results = tfnet.return_predict(frame)#return obj as frame, whichever obj is detected
        for color, result in zip(colors, results):#loop for getting 1 color for every result
            tl = (result['topleft']['x'], result['topleft']['y'])#mark for top right
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']#passing name of the label
            confidence = result['confidence']#precent of confidence, how much percent algo is confident 
            text = '{}: {:.0f}%'.format(label, confidence * 100)#to show percentage of confidence in label
            frame = cv2.rectangle(frame, tl, br, color, 5)#to form a rectangle around our frame tl br and color of ractangle and line with will be 5 ie last passing value
            frame = cv2.putText(#putting text on frame 
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)#first value we will pass is text, tl= top left, font name "font hershey complex", 1 is font size,(0,0,0)is color of font ie black , 2 is width
        cv2.imshow('frame', frame)#coming our of for loop check indent, showing output with the name frame window 
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))#printing FPS, (:.1f) is format float,1/(currenttime / start_time) and we want 1 decimal place  
    if cv2.waitKey(1) & 0xFF == ord('q'):#quit key = q make sure you hit q to quit and not ctrl +c or some random stuff 
        break

capture.release()# turnoff webcam
cv2.destroyAllWindows()