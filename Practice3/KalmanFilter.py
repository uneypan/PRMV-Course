# -*- coding:utf-8 -*-

import numpy as np


class KalmanFilter(object):
    
    def __init__(self, Q1=0.01, Q2=0.001, R1=0.06, P1=0.01, P2=0.01):
        '''
        Kalman Filter for beat tracking. 
        Parameters:
        x: initial state vector ex. [position, period]
        P1: beat time initial uncertainty convariance
        P2: beat period initial uncertainty convariance
        R1: beat time measurement variance
        Q1: beat time estimate uncertainty convariance
        Q2: beat period estimate uncertainty convariance
        '''
        self.A = np.asanyarray([[1,1],[0,1]], dtype=np.float32) 
        self.H = np.asanyarray([[1,0]], dtype=np.float32)
        self.Q = np.asanyarray([[Q1,0],[0,Q2]], dtype=np.float32)
        self.R = np.asanyarray(R1, dtype=np.float32)
        self.P = np.asanyarray([[P1,0],[0,P2]], dtype=np.float32)
        self.x = np.asanyarray([0,1], dtype=np.float32)
        self.z = np.asanyarray(0, dtype=np.float32)
    
    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x
    
    def update(self):
        self.K = np.dot(np.dot(self.P, self.H.T), 
            np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R))
        self.x = self.x + np.dot(self.K, (self.z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(np.dot(self.K, self.H), self.P)
        return self.x
    
    def run(self, z):
        self.z = z
        self.predict()
        self.update()
    

if __name__ == "__main__":
    
    z = [1,2,3,4,5,6,7,8,9,10]
    kf = KalmanFilter()
    kf.x = np.asanyarray([0,1.5])
    for i in z:
        kf.run(z=i)
        print(kf.x) 