import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import time, math

class ParticleFilter:
    def __init__(self, N, target_hist, max_x, max_y):
        self.N = N
        self.hist = target_hist
        self.dx = 5
        self.dy = 5
        self.mx = max_x
        self.my = max_y



    def state_init(self, max_x, max_y):
        x = np.random.randint(max_x, size=self.N)
        y = np.random.randint(max_y, size=self.N)
        s = np.concatenate( (x, y) )
        s = np.reshape(s, (2, self.N))
        weights = np.full( (1, self.N), 1/self.N ) 
        return s, weights



    def predict(self, s, max_x, max_y):
        s[0,:] = s[0,:] + self.dx
        s[1,:] = s[1,:] + self.dy
        idx_x = np.where(s[0,:] >= max_x)
        idx_y = np.where(s[1,:] >= max_y)
        s[0, idx_x] = max_x-1
        s[1, idx_y] = max_y-1
        return s
        


    def update(self, s, Hx, Hy, frame):
        #imag_mask = frame[sq[1]:sq[1]+sq[3], sq[0]:sq[0]+sq[2], :]
        d = np.zeros(self.N)
        
        print("update")

        #for i in range(0, self.N):
        msk_idx = np.arange(s[0,:] - Hx,s[0:] + Hx)
        msk_idy = np.arange(s[1,:] - Hy,s[1,:] + Hy)
        idx_x = np.where(msk_idx >= self.mx)
        idx_y = np.where(msk_idy >= self.my)
        msk_idx[idx_x] = self.mx-1
        msk_idy[idx_y] = self.my-1
        idx_x = np.where(msk_idx == 0)
        idx_y = np.where(msk_idy == 0)            
        msk_idx[idx_x] = 1
        msk_idy[idx_y] = 1

        imag_mask = frame[int(msk_idx[0]):int(msk_idx[-1]), int(msk_idy[0]):int(msk_idy[-1]) , :]
        hist_r, bins_r = np.histogram(imag_mask[:,:,0], bins=50, range=(0, 255), density=True)
        
        # plt.plot(hist_r), plt.title('Red')
        # plt.show()

        H1 = np.sum(self.hist)/50
        H2 = np.sum(hist_r)/50
        H = np.sum(np.multiply(self.hist, hist_r))

        d[i] = math.sqrt( 1 -  H / (50 * math.sqrt(H1*H2)) )
        print(d)
        return

