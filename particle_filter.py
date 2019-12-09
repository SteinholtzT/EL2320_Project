import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import time, math

class ParticleFilter:
    def __init__(self, N, target_hist, max_x, max_y, bins, dt):
        self.N = N
        self.hist = target_hist
        self.dx = 5
        self.dy = 5
        self.mx = max_x
        self.my = max_y
        self.bins = bins
        self.dt = dt



    def state_init(self):
        #x = np.random.randint(self.mx, size=self.N)
        #y = np.random.randint(self.my, size=self.N)
        #s = np.concatenate( (x, y) )
        #s = np.reshape(s, (2, self.N))
        weights = np.full( (1, self.N), 1/self.N ) 
        
        #State vector
        s = np.array([  [np.random.randint(self.mx, size=self.N)],  # x
                        [np.random.randint(self.my, size=self.N)],  # y 
                        [np.random.randint(-10, 10, size=self.N)],  # dx
                        [np.random.randint(-10, 10, size=self.N)]    ])    # dy
        s = np.reshape(s, (4, self.N))
        s = s.astype(int)

        return s, weights



    def predict(self, s):
        x = np.copy(s[0,:])
        y = np.copy(s[1,:])
        #Predict new 
        s[0,:] = np.sum([s[0,:], s[2, :]], axis=0)
        s[1,:] = np.sum([s[1,:], s[3, :]], axis=0)
        s[2,:] = np.divide( np.subtract(s[0,:], x), self.dt ) 
        s[3,:] = np.divide( np.subtract(s[1,:], y), self.dt ) 
        idx_x = np.where(s[0,:] >= self.mx)
        idx_y = np.where(s[1,:] >= self.my)
        s[0, idx_x] = self.mx-1
        s[1, idx_y] = self.my-1
        idx_x = np.where(s[0,:] <= 0)
        idx_y = np.where(s[1,:] <= 0)
        s[0, idx_x] = 1
        s[1, idx_y] = 1
        
        s = s.astype(int)
        print(s)
        return s
        


    def update(self, s, Hx, Hy, frame):
        d = np.zeros(self.N)
        
        for i in range(0, self.N):
            msk_idx = np.arange(s[0,i] - Hx,s[0,i] + Hx)
            msk_idy = np.arange(s[1,i] - Hy,s[1,i] + Hy)
            idx_x = np.where(msk_idx >= self.mx)
            idx_y = np.where(msk_idy >= self.my)
            msk_idx[idx_x] = self.mx-1
            msk_idy[idx_y] = self.my-1
            idx_x = np.where(msk_idx <= 0)
            idx_y = np.where(msk_idy <= 0)            
            msk_idx[idx_x] = 1
            msk_idy[idx_y] = 1

            imag_mask = frame[int(msk_idx[0]):int(msk_idx[-1]), int(msk_idy[0]):int(msk_idy[-1]) , :]
            hist_r, bins_r = np.histogram(imag_mask[:,:,0], bins=self.bins, range=(0, 255), density=True)
            
            H = np.sum( np.sqrt( np.multiply(self.hist, hist_r) ) )
            d[i] = math.sqrt(1-H)
        
        return

