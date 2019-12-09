import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import time, math

class ParticleFilter:
    def __init__(self, N, target_hist, max_x, max_y, bins, dt, Q):
        self.N = N
        self.hist = target_hist
        self.mx = max_x
        self.my = max_y
        self.bins = bins
        self.dt = dt
        self.Q = Q
        self.weights = np.full( (1, self.N), 1/self.N ) 




    def state_init(self):
     
        #State vector
        s = np.array([  [np.random.randint(self.mx, size=self.N)],  # x
                        [np.random.randint(self.my, size=self.N)],  # y 
                        [np.random.randint(-10, 10, size=self.N)],  # dx
                        [np.random.randint(-10, 10, size=self.N)]    ])    # dy
        s = np.reshape(s, (4, self.N))
        s = s.astype(int)
        return s

    def predict(self, s, Hx, Hy):
        x = np.copy(s[0,:])
        y = np.copy(s[1,:])
        #Predict new 
        s[0,:] = np.sum([s[0,:], s[2, :]], axis=0)
        s[1,:] = np.sum([s[1,:], s[3, :]], axis=0)
        s[2,:] = np.divide( np.subtract(s[0,:], x), self.dt ) 
        s[3,:] = np.divide( np.subtract(s[1,:], y), self.dt ) 
        s = s.astype(int)
        
        idx_xmax = np.where(s[0,:]+Hx >= self.mx)
        idx_ymax = np.where(s[1,:]+Hy >= self.my)
        s[0, idx_xmax] = np.random.randint(self.mx, size=1)
        s[1, idx_ymax] = np.random.randint(self.my, size=1)
        s[2, idx_xmax] = np.random.randint(-10, 10, size=1)
        s[3, idx_ymax] = np.random.randint(-10, 10, size=1)

        idx_xmin = np.where(s[0,:]-Hx <= 0)
        idx_ymin = np.where(s[1,:]-Hy <= 0)
        s[0, idx_xmin] = np.random.randint(self.mx, size=1)
        s[1, idx_ymin] = np.random.randint(self.my, size=1)
        s[2, idx_xmin] = np.random.randint(-10, 10, size=1)
        s[3, idx_ymin] = np.random.randint(-10, 10, size=1)
        s = s.astype(int)
        print(s)
        return s
        


    def update(self, s, Hx, Hy, frame):
        dt = np.zeros(self.N)
                d = np.zeros(self.N)
        prob = np.zeros(self.N)

        for i in range(0, self.N):
            msk_idx = np.arange(s[0,i] - Hx,s[0,i] + Hx)
            msk_idy = np.arange(s[1,i] - Hy,s[1,i] + Hy)
            # idx_x = np.where(msk_idx >= self.mx)
            # idx_y = np.where(msk_idy >= self.my)
            # msk_idx[idx_x] = self.mx
            # msk_idy[idx_y] = self.my
            # idx_x = np.where(msk_idx <= 0)
            # idx_y = np.where(msk_idy <= 0)            
            # msk_idx[idx_x] = 1
            # msk_idy[idx_y] = 1

            #print(msk_idx[0], msk_idx[-1], msk_idy[0], msk_idy[-1])

            imag_mask = frame[int(msk_idx[0]):int(msk_idx[-1]), int(msk_idy[0]):int(msk_idy[-1]) , :]
            hist_r, bins_r = np.histogram(imag_mask[:,:,0], bins=self.bins, range=(0, 257), density=True)
        
        # plt.plot(hist_r), plt.title('Red')
        # plt.show()
    
            H = np.sum(np.sqrt(np.multiply(self.hist, hist_r)))

            d[i] = math.sqrt( 1 -  H)
            prob[i] = (1/(self.Q)*math.sqrt(2*math.pi))*math.exp(-d[i]/(2*self.Q**2))
            

        self.weights = np.multiply(self.weights, prob)
        resmpling_fac = 1/(self.N*np.sum(np.multiply(self.weights,self.weights)))
        
  
         
            Ht = np.sum( np.sqrt( np.multiply(self.hist, hist_r) ) )
            dt[i] = math.sqrt(1-H)
       
        if resampling_fac>1:
            resampling()
        
        return
    
    
    
    
    def resampling(self):
        CDF = cumsum(self.weights)
        r0 = np.random.random_sample()/self.N
    
    for m in range(self.N):  
        
        i = CDF>=(r0+(m-1)/self.N)
        print(i)
        #S(:, m)=S_bar(:, i)
    
    
    #S(4, :) = 1/M