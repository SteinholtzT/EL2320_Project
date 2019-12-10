import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import time, math

class ParticleFilter:
    def __init__(self, N, target_hist, max_x, max_y, bins, dt, Q, R):
        self.N = N
        self.hist = target_hist
        self.mx = max_x
        self.my = max_y
        self.bins = bins
        self.dt = dt
        self.R = R
        self.Q = Q
        self.weights = np.full( (1, self.N), 1/self.N )
        self.s = None 




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
        s[0, idx_xmax] = np.random.randint(Hx, self.mx-Hx, size=len(idx_xmax))
        s[1, idx_ymax] = np.random.randint(Hx, self.my-Hy, size=len(idx_ymax))

        s[2, idx_xmax] = 0#np.random.randint(-10, 10, size=len(idx_xmax))
        s[3, idx_ymax] = 0#np.random.randint(-10, 10, size=len(idx_ymax))



        idx_xmin = np.where(s[0,:]-Hx < 0)
        idx_ymin = np.where(s[1,:]-Hy < 0)
        s[0, idx_xmin] = np.random.randint(Hx, self.mx-Hx, size=len(idx_xmin))
        s[1, idx_ymin] = np.random.randint(Hy, self.my-Hy, size=len(idx_ymin))

        s[2, idx_xmin] = 0#np.random.randint(-10, 10, size=len(idx_xmin))
        s[3, idx_ymin] = 0#np.random.randint(-10, 10, size=len(idx_ymin))

        s = s.astype(int)

        self.s = s
        return self.s
        


    def update(self, Hx, Hy, frame):
        #dt = np.zeros(self.N)
        # d_r = np.zeros(self.N)
        # d_g = np.zeros(self.N)
        # d_b = np.zeros(self.N)
        prob = np.zeros(self.N)

        for i in range(0, self.N):
            msk_idx = np.arange(self.s[0,i] - Hx,self.s[0,i] + Hx)
            msk_idy = np.arange(self.s[1,i] - Hy,self.s[1,i] + Hy)


            imag_mask = frame[int(msk_idx[0]):int(msk_idx[-1]), int(msk_idy[0]):int(msk_idy[-1]) , :]
            hist_b, bins_r = np.histogram(imag_mask[:,:,0], bins=self.bins, range=(0, 255), density=True)
            hist_g, bins_r = np.histogram(imag_mask[:,:,1], bins=self.bins, range=(0, 255), density=True)
            hist_r, bins_r = np.histogram(imag_mask[:,:,2], bins=self.bins, range=(0, 255), density=True)
            
            
            H_r = np.sum(np.sqrt(np.multiply(self.hist, hist_r)))
            H_g = np.sum(np.sqrt(np.multiply(self.hist, hist_g)))
            H_b = np.sum(np.sqrt(np.multiply(self.hist, hist_b)))

            d_r = math.sqrt( 1 -  H_r)
            d_g = math.sqrt( 1 -  H_g)
            d_b = math.sqrt( 1 -  H_b)

            prob_r = (1/(self.R)*math.sqrt(2*math.pi))*math.exp(-d_r/(2*self.R**2))
            prob_g = (1/(self.Q)*math.sqrt(2*math.pi))*math.exp(-d_g/(2*self.R**2))
            prob_b = (1/(self.Q)*math.sqrt(2*math.pi))*math.exp(-d_b/(2*self.R**2))

            prob[i] = prob_r*prob_g*prob_b
        
            
        self.weights = np.multiply( self.weights, prob)
        self.weights = np.divide(self.weights, np.sum(self.weights))

        # print('max = ' ,max(self.weights[0]))
        # print('min = ' ,min(self.weights[0]))
        # print(' ')

        resampling_fac = 1/(self.N*np.sum(np.multiply(self.weights,self.weights)))
        
        s_esti = np.zeros(2)
        s_esti[0] = np.sum( np.multiply(self.weights, self.s[0, :] ))
        s_esti[1] = np.sum( np.multiply(self.weights, self.s[1, :] ))

        idx = np.array([   [s_esti[0]-Hx, s_esti[0]+Hx],
                            [s_esti[1]-Hy, s_esti[1]+Hx] ])
        idx = idx.astype(int)
        # idy = np.arange(s_esti[1]-Hy, s_esti[1]+Hy) 
        # idx = idx.astype(int)
        # idy = idy.astype(int)

        #print(max(self.weights[0]))
        #print(resampling_fac)
        if resampling_fac<0.95:
            #print('Resampling')
            self.s = self.resampling(self.s)

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

