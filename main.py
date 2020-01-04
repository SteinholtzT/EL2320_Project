# importing libraries 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import time
from particle_filter import ParticleFilter

#Partile Filter Parameters
N = 100
bins = 180
i = 1
dt = 1/(30)
Q = 0.4 # measurment noise
R = 0.03   # process noise
predict_n = 1 # number of frames before predict

#While loop Parameters
captured = False
sample = 0

## Choose which video to play ##

# cap = cv2.VideoCapture('Cocacola.mp4')
# cap = cv2.VideoCapture('result1.mp4')
# cap = cv2.VideoCapture('result2.mp4')
# cap = cv2.VideoCapture('result3.mp4')

cap = cv2.VideoCapture(0) # Live camera feed



if (cap.isOpened()== False):  
  print("Error opening video file, please") 

t0 = time.time()


while(cap.isOpened()): 
    t2 = time.time()    
    ret, image = cap.read()

    if ret == True:
        frame = cv2.resize(image,(640, 360), interpolation = cv2.INTER_LINEAR)
        max_x, max_y, z = np.shape(frame)
        particle_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if sample!=predict_n and captured == True:
            s = PF.predict(sq[3]/2, sq[2]/2)
            particle_frame[s[0,:], s[1,:] ] = [0, 255, 0]
            idx, idy = PF.update(sq[3]/2, sq[2]/2, hsv)
            frame[ idx[0], idy[0]:idy[-1] ] = [255, 0, 0]
            frame[ idx[-1], idy[0]:idy[-1] ] = [255, 0, 0]
            frame[ idx[0]:idx[-1], idy[0] ] = [255, 0, 0] 
            frame[ idx[0]:idx[-1], idy[-1] ] = [255, 0, 0]
            sample += 1

        elif sample == predict_n and captured == True:
            particle_frame[s[0,:], s[1,:] ] = [0, 255, 0]

            frame[ idx[0], idy[0]:idy[-1] ] = [255, 0, 0]
            frame[ idx[-1], idy[0]:idy[-1] ] = [255, 0, 0]
            frame[ idx[0]:idx[-1], idy[0] ] = [255, 0, 0]
            frame[ idx[0]:idx[-1], idy[-1] ] = [255, 0, 0]
            sample = 0

        dst = cv2.addWeighted(frame,0,particle_frame,0.9,0)
        cv2.imshow('Frame', dst)

  # Press C to choose object to track
        if cv2.waitKey(25) & 0xFF == ord('c'):
            captured = True
            sq = cv2.selectROI('Choose object', hsv)
            cv2.destroyWindow('Choose object')
           
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            imag_mask = hsv[sq[1]:sq[1]+sq[3], sq[0]:sq[0]+sq[2]]

            hist_h = cv2.calcHist([imag_mask[:,:,0]],[0],None,[bins],[0,360])
            hist_s = cv2.calcHist([imag_mask[:,:,1]],[0],None,[bins],[0,360])
            hist_v = cv2.calcHist([imag_mask[:,:,2]],[0],None,[bins],[0,360])

            print("Histogram made")

            # Initialise Particle Filter
            PF = ParticleFilter(N, hist_h, hist_s, hist_v, max_x, max_y, bins, dt, Q, R)
            s = PF.state_init()
            particle_frame[s[0,:], s[1,:] ] = [0, 255, 0]
        t3 = time.time() 

        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    
    else:  
        break

t1 = time.time()
print('Total time: ', t1-t0)
cap.release() 
cv2.destroyAllWindows()