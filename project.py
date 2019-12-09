# importing libraries 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
   
# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('Cocacola.mp4')
# Check if camera opened successfully 
if (cap.isOpened()== False):  
  print("Error opening video  file") 
   
# Read until video is completed 
while(cap.isOpened()): 
      
  # Capture frame-by-frame 
  ret, frame = cap.read() 


  color_upper = np.array([250 , 40, 40])
  color_lower = np.array([230 , 10 , 10])
  index = np.where((frame < color_upper) & (frame > color_lower))

  cord = list(zip(index[0], index[1])) 
  cord = np.array(cord)

  #frame[cord[:,0], cord[:,1], :] = [0, 0, 0]

  tracking_matrix = np.zeros(frame.shape)


  if ret == True: 
    # Display the resulting frame 
    cv2.imshow('Frame', frame)
    
    # Press C
    if cv2.waitKey(25) & 0xFF == ord('c'):
      sq = cv2.selectROI('Choose object', frame)
      cv2.destroyWindow('Choose object')

      imag_mask = frame[sq[1]:sq[1]+sq[3], sq[0]:sq[0]+sq[2], :]

      hist_b = np.histogram(imag_mask[:,:,0], bins = np.arange(257))
      hist_g = np.histogram(imag_mask[:,:,1], bins = np.arange(257))
      hist_r = np.histogram(imag_mask[:,:,2], bins = np.arange(257))

      plt.subplot(131), plt.plot(hist_r[0]), plt.title('red')
      plt.subplot(132), plt.plot(hist_g[0]), plt.title('green')
      plt.subplot(133), plt.plot(hist_b[0]), plt.title('blue')

      plt.show()

      
      

       
   
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





