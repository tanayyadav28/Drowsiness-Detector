# importing the necessary libraries

import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
                                                   

'''
- We use haar cascade classifier to detect the facial area and the eyes, 
- The haar classifier uses haar kernels to detect the features and then uses
AdaBoost to cut down features
- Then these cut down features are put through attentional cascade to speed up 
the process by creating 2 sets of filters and processing using the set 2 only if 
set 1 returns true.
- The cascade classifier then gives us the coordinates for the concerned feature.
- We then process the sub patch from the frame and use a keras model to classify 
the eyes as : ['closed', 'open']

- This program will save the image of the person sleeping as a proof and display
the time which they slept at a time and sum of all the times they have slept.
'''


mixer.init()
sound = mixer.Sound('alarm.wav')                                                    # to play the alarm sound if drowsy

face = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')                # to identify the face
leye = cv2.CascadeClassifier('haar/haarcascade_lefteye_2splits.xml')                # to identify the left eye
reye = cv2.CascadeClassifier('haar/haarcascade_righteye_2splits.xml')               # to identify the right eye

model = load_model('models/cnncat2.h5')                                             # loading the keras model
path = os.getcwd()
cap = cv2.VideoCapture(0)                                                           # live camera feed capture
font = cv2.FONT_HERSHEY_DUPLEX

if not os.path.exists(os.path.join(path, 'images')):
    os.makedirs(os.path.join(path, 'images'))


score = 0
worst_scores = [0]
thicc = 2
closed = 0
write = 0
lpred, rpred = [1], [1]
unatt_t = [0]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                  # converting BGR (3 channels) to 1 channel (faster processing)
    
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-40), (140,height), (0,0,0), thickness=cv2.FILLED)
 
    for (x,y,w,h) in right_eye:
        
        r_eye = frame[y : y+h, x : x+w]                                             # patch containing the right eye
        
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_temp = r_eye       
        r_eye = cv2.resize(r_eye, (24,24))
        r_eye = r_eye/255                                                           # [0, 255] -> [0.0, 1.0]
        r_eye =  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        
        xpred = model.predict(r_eye) 
        rpred = np.argmax(xpred, axis=1)                                            # get the predicted class
        
        if rpred[0] == 1:
            lbl = 'Open' 
            cv2.imwrite('images/open_reye.png', r_temp)                             # visualize the identified features 
        if rpred[0] == 0:
            lbl = 'Closed'
            cv2.imwrite('images/close_reye.png', r_temp)
        
        break

    for (x,y,w,h) in left_eye:
        
        l_eye = frame[y : y+h, x : x+w]
        
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)  
        l_temp = l_eye
        l_eye = cv2.resize(l_eye, (24,24))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        
        xpred = model.predict(l_eye) 
        lpred = np.argmax(xpred, axis=1)
        
        if lpred[0] == 1:
            lbl = 'Open'  
            cv2.imwrite('images/open_leye.png', l_temp) 
        if lpred[0] == 0:
            lbl = 'Closed'
            cv2.imwrite('images/close_leye.png', l_temp)
        
        break
    
    if  rpred[0] == 0 and lpred[0] == 0:                                            # both eyes are closed
        color_b = (50, 50, 230)
        thick_b = 4
        if closed == 0:
            s_t = time.time()
            closed = 1                                                                                            
        score = score + 1


    else: 
        color_b = (50, 230, 50)  
        thick_b = 1 
        score = score - 1
        if closed == 1:
            e_t = time.time()
            time_s = e_t - s_t
            if time_s > 2:
                unatt_t.append(time_s)
            closed = 0

    
    for (x,y,w,h) in faces:                                                         # create a rectangle around the identified face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_b, thick_b)                  # colors it red or green based on close/open eyes
    
    
    if score < 0:                                                                   # ensuring positive score
        score=0   
    
    cv2.putText(frame, f'Score: {str(100 - score)}', (10,height-15), font, 0.7, (255,255,255), 1, cv2.LINE_AA)
    
    if score > max(worst_scores):
        worst_scores.append(score)

    if score > 35:                                                                  # person asleep for more than 15 frames
        write = 1
        if score >= max(worst_scores):
            temp_frame = frame
          
        try:                                                                        # to play the sound only if it is not playing
            sound.play()            
        except:  
            pass
        
        if thicc < 24:
            thicc = thicc + 4
        
        else:
            thicc = thicc - 4
            
            if thicc < 2:
                thicc = 2
        
        cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thicc) 
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):                                           # quitting the detection test and printing the results
        
        print('-'*150)
        print('* You were unattentive at once for more than {:.3f}s'.format(max(unatt_t)))
        print('* You were unattentive for a total of {:.3f}s'.format(sum(unatt_t)))
        if write == 1:                                                              # saving the image once alarm beeps
            cv2.imwrite(os.path.join(path,'images/slept.jpg'), temp_frame)
            print('* Image saved at: {}'.format(os.path.join(path, 'images/slept.jpg')))
        print('-'*150)
        break

cap.release()
cv2.destroyAllWindows()