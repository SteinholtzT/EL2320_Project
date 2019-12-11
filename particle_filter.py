import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import time, math

class ParticleFilter:
    def __init__(self, N, target_hist_h, target_hist_s, target_hist_v, max_x, max_y, bins, dt, Q, R):
        self.N = N
        self.hist_th = target_hist_h
        self.hist_ts = target_hist_s
        self.hist_tv = target_hist_v
        self.mx = max_x
        self.my = max_y
        self.bins = bins
        self.dt = dt
        self.R = R
        self.Q = Q
        self.weights = np.full( (1, self.N), 1/self.N )
        self.s = None 
        self.meanp = 0




    def state_init(self):
     
        #State vector
        self.s = np.array([  [np.random.randint(self.mx, size=self.N)],  # x
                        [np.random.randint(self.my, size=self.N)],  # y 
                        [np.zeros(self.N)],  # dx
                        [np.zeros(self.N)]    ])    # dy

        self.s = np.reshape(self.s, (4, self.N))
        self.s = self.s.astype(int)
        return self.s


    def predict(self, Hx, Hy):
        s = self.s
        x = np.copy(self.s[0,:])
        y = np.copy(self.s[1,:])

        #Predict new 

        s[0,:] = np.sum([s[0,:], s[2, :]], axis=0) + np.random.normal(0, self.R, self.N)
        s[1,:] = np.sum([s[1,:], s[3, :]], axis=0) + np.random.normal(0, self.R, self.N)
        s[2,:] = np.divide( np.subtract(s[0,:], x), self.dt ) + np.random.normal(0, self.R, self.N)
        s[3,:] = np.divide( np.subtract(s[1,:], y), self.dt ) + np.random.normal(0, self.R, self.N)

        s = s.astype(int)
        
        idx_xmax = np.where(s[0,:]+Hx > self.mx-1)
        idx_ymax = np.where(s[1,:]+Hy > self.my-1)
        s[0, idx_xmax] = np.random.randint(Hx, self.mx-Hx, len(idx_xmax[0]))
        s[1, idx_ymax] = np.random.randint(Hx, self.my-Hy, len(idx_ymax[0]))

        s[2, idx_xmax] = 0#np.random.randint(-10, 10, size=len(idx_xmax))
        s[3, idx_ymax] = 0#np.random.randint(-10, 10, size=len(idx_ymax))



        idx_xmin = np.where(s[0,:]-Hx < 0)
        idx_ymin = np.where(s[1,:]-Hy < 0)
        s[0, idx_xmin] = np.random.randint(Hx, self.mx-Hx, len(idx_xmin[0]))
        s[1, idx_ymin] = np.random.randint(Hy, self.my-Hy, len(idx_ymin[0]))

        s[2, idx_xmin] = 0#np.random.randint(-10, 10, size=len(idx_xmin))
        s[3, idx_ymin] = 0#np.random.randint(-10, 10, size=len(idx_ymin))

        s = s.astype(int)

        self.s = s
        return self.s
        


    def update(self, Hx, Hy, frame):
        d_r = np.zeros(self.N)
        d_g = np.zeros(self.N)
        d_b = np.zeros(self.N)
        prob = np.zeros(self.N)

        for i in range(0, self.N):
            msk_idx = np.arange(self.s[0,i] - Hx,self.s[0,i] + Hx)
            msk_idy = np.arange(self.s[1,i] - Hy,self.s[1,i] + Hy)

            imag_mask = frame[int(msk_idx[0]):int(msk_idx[-1]), int(msk_idy[0]):int(msk_idy[-1]) , :]
            # imag_mask = hsv[int(msk_idx[0]):int(msk_idx[-1]), int(msk_idy[0]):int(msk_idy[-1]), :]
            hist_h, bins_r = np.histogram(imag_mask[:,:,0], bins=self.bins, range=(0, 360), density=True)
            hist_s, bins_r = np.histogram(imag_mask[:,:,1], bins=self.bins, range=(0, 360), density=True)
            hist_v, bins_r = np.histogram(imag_mask[:,:,2], bins=self.bins, range=(0, 360), density=True)
            
            H_h = np.sum(np.sqrt(np.multiply(self.hist_th, hist_h)))
            H_s = np.sum(np.sqrt(np.multiply(self.hist_ts, hist_s)))
            H_v = np.sum(np.sqrt(np.multiply(self.hist_tv, hist_v)))

            d_h = math.sqrt( 1 -  H_h)
            d_s = math.sqrt( 1 -  H_s)
            d_v = math.sqrt( 1 -  H_v)

            prob_h = (1/(self.Q**5)*math.sqrt(2*math.pi))*math.exp(-d_h/(2*self.Q**2))
            prob_s = (1/(self.Q**5)*math.sqrt(2*math.pi))*math.exp(-d_s/(2*self.Q**2))
            prob_v = (1/(self.Q**5)*math.sqrt(2*math.pi))*math.exp(-d_v/(2*self.Q**2))

            prob[i] = prob_h*prob_s*prob_v
        
        
        self.weights = np.multiply( self.weights, prob)
        self.weights = np.divide(self.weights, np.sum(self.weights))


        resampling_fac = 1/(self.N*np.sum(np.multiply(self.weights,self.weights)))
        if resampling_fac<1: 
            self.s = self.resampling(self.s)
        
        s_esti = np.zeros(2)
        s_esti[0] = np.sum( np.multiply(self.weights, self.s[0, :] ))
        s_esti[1] = np.sum( np.multiply(self.weights, self.s[1, :] ))
        

        idx = np.array([[s_esti[0]-Hx, s_esti[0]+Hx],
                        [s_esti[1]-Hy, s_esti[1]+Hy]])

        idx[0, np.where(idx[0,:]<0)] = 1
        idx[0, np.where(idx[0,:]>=self.mx)] = self.mx - 1
        idx[1, np.where(idx[1,:]<0)] = 1
        idx[1, np.where(idx[1,:]>=self.my)] = self.my - 1
            
        idx = idx.astype(int)
        return idx
    
    
    

    def resampling(self, s_bar):
        CDF = np.cumsum(self.weights)
        r0 = np.random.random_sample()/self.N
        s = np.zeros(s_bar.shape)

        for m in range(self.N):  
            i = np.where(CDF>=((r0+(m-1))/self.N))
            s[:, m]=s_bar[:, i[0][0]]
        self.weights = np.full( (1, self.N), 1/self.N )
        return s

