# importing libraries 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
   
from particle_filter import ParticleFilter



#Partile Filter Parameters
N = 300
bins = 180*2
i = 1
dt = 1/(30)
Q = 0.3 # process noise
R = 0.2 # measurment noise
predict_n = 1 # number of frames before predict

#While loop Parameters
captured = False
sample = 0

# Create a VideoCapture object and read from input file 

# cap = cv2.VideoCapture('Cocacola.mp4')
# cap = cv2.VideoCapture('boll.mp4')
# cap = cv2.VideoCapture('cola1.mp4')
# cap = cv2.VideoCapture('cola2.mp4')
# cap = cv2.VideoCapture('tim.mp4')

cap = cv2.VideoCapture(0)

#url = 'http://192.168.0.100:4747/video'
#cap = cv2.VideoCapture(url)


# Check if camera opened successfully 
if (cap.isOpened()== False):  
  print("Error opening video  file") 



# Read until video is completed 
while(cap.isOpened()): 
      
  # Capture frame-by-frame 
    ret, frame = cap.read()
    max_x, max_y, z = np.shape(frame)
    particle_frame = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if ret == True:
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

  # Press C
        if cv2.waitKey(25) & 0xFF == ord('c'):
            captured = True
            sq = cv2.selectROI('Choose object', hsv)
            cv2.destroyWindow('Choose object')

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # imag_mask = frame[sq[1]:sq[1]+sq[3], sq[0]:sq[0]+sq[2]]
            imag_mask = hsv[sq[1]:sq[1]+sq[3], sq[0]:sq[0]+sq[2]]
            # hist_h, bins_r = np.histogram(imag_mask[:,:,0], bins=bins, range=(0, 360), density=True)
            # hist_s, bins_b = np.histogram(imag_mask[:,:,1], bins=bins, range=(0, 360), density=True)
            # hist_v, bins_g = np.histogram(imag_mask[:,:,2], bins=bins, range=(0, 360), density=True)

            hist_h = cv2.calcHist([imag_mask[:,:,0]],[0],None,[bins],[0,360])
            hist_s = cv2.calcHist([imag_mask[:,:,1]],[0],None,[bins],[0,360])
            hist_v = cv2.calcHist([imag_mask[:,:,2]],[0],None,[bins],[0,360])

            print("Histogram made")

            # Initialise Particle Filter
            PF = ParticleFilter(N, hist_h, hist_s, hist_v, max_x, max_y, bins, dt, Q, R)
            s = PF.state_init()
            particle_frame[s[0,:], s[1,:] ] = [0, 255, 0]
        
        # Press Q on keyboard to  exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    
  # Break the loop 
    else:  
        break
   
# When everything done, release  
# the video capture object 
cap.release() 
   
# Closes all the frames 
cv2.destroyAllWindows()