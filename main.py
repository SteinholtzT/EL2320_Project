# importing libraries 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
   
from particle_filter import ParticleFilter



#Partile Filter Parameters
N = 100
bins = 50
i = 1
dt = 1/(25*10**-3)
Q = 0.2
R = 2



#While loop Parameters
captured = False


# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('Cocacola.mp4')
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

    if ret == True:
        if captured == True:
            s = PF.predict(sq[3]/2, sq[2]/2)
            particle_frame[s[0,:], s[1,:] ] = [0, 255, 0]
            idx, idy = PF.update(sq[3]/2, sq[2]/2, gray)
            frame[ idx[0], idy[0]:idy[-1] ] = [255, 0, 0]
            frame[ idx[-1], idy[0]:idy[-1] ] = [255, 0, 0]
            frame[ idx[0]:idx[-1], idy[0] ] = [255, 0, 0]
            frame[ idx[0]:idx[-1], idy[-1] ] = [255, 0, 0]




        dst = cv2.addWeighted(frame,0.1,particle_frame,0.9,0)
        cv2.imshow('Frame', dst)

  # Press C
        if cv2.waitKey(25) & 0xFF == ord('c'):
            captured = True
            sq = cv2.selectROI('Choose object', gray)
            cv2.destroyWindow('Choose object')

            imag_mask = gray[sq[1]:sq[1]+sq[3], sq[0]:sq[0]+sq[2]]
            print(np.shape(imag_mask))
            hist_r, bins_r = np.histogram(imag_mask[:,:], bins=bins, range=(0, 255), density=True)
            #hist_b, bins_b = np.histogram(imag_mask[:,:,1], bins=bins, range=(0, 255), density=True)
            #hist_g, bins_b = np.histogram(imag_mask[:,:,2], bins=bins, range=(0, 255), density=True)
            print("Histogram made")

            # Initialise Particle Filter
            PF = ParticleFilter(N, hist_r, max_x, max_y, bins, dt, Q, R)
            s = PF.state_init()
            particle_frame[s[0,:], s[1,:] ] = [0, 255, 0]


            # plt.subplot(231), plt.plot(hist_r), plt.title('Red')
            # plt.subplot(232), plt.plot(hist_g), plt.title('Green')
            # plt.subplot(233), plt.plot(hist_b), plt.title('Blue')
            # plt.show()




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