import numpy as np 

class particle_filter:
    def __init__(self, N, R, Q, part):
        
        # Number of particles
        self.N = N
        
        # Covariance matrix motion model
        self.R = R 

        # Covariance matrix measurment model
        self.Q = Q

        # Particles
        self.particles = 

    def resample(self, weights):
        weights = 0

    def predict(self, particles, measurment):
        cov_Q = np.random.normal(0, Q, 1000)
        

        error = 


