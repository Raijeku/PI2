import numpy as np
import pandas as pd
from qiskit import Aer, IBMQ, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from math import pi
from sklearn.preprocessing import normalize, scale

def preprocess(points,type='angle'):
    if type == 'angle': points = scale(points)
    elif type == 'probability': points = normalize(points)
    return points

def distance(x,y,backend,type='angle'):
    if type == 'angle':
        complexes_x = x[0] + 1j*x[1]
        complexes_y= y[0] + 1j*y[1]
        theta_1 = np.angle(complexes_x)
        theta_2 = np.angle(complexes_y)
    
        # create Quantum Register called "qr" with 3 qubits
        qr = QuantumRegister(3, name="qr")
        # create Classical Register called "cr" with 5 bits
        cr = ClassicalRegister(3, name="cr")

        # Creating Quantum Circuit called "qc" involving your Quantum Register "qr"
        # and your Classical Register "cr"
        qc = QuantumCircuit(qr, cr, name="k_means")

        qc.h(qr[0])
        qc.h(qr[1])
        qc.h(qr[2])
        qc.u3(theta_1, pi, pi, qr[1])
        qc.u3(theta_2, pi, pi, qr[2])
        qc.cswap(qr[0], qr[1], qr[2])
        qc.h(qr[0])

        qc.measure(qr[0], cr[0])
        qc.reset(qr)
    elif type == 'probability':
        # create Quantum Register called "qr" with 3 qubits
        qr = QuantumRegister(3, name="qr")
        # create Classical Register called "cr" with 5 bits
        cr = ClassicalRegister(3, name="cr")

        # Creating Quantum Circuit called "qc" involving your Quantum Register "qr"
        # and your Classical Register "cr"
        qc = QuantumCircuit(qr, cr, name="k_means")
        qc.initialize(x,1)
        qc.initialize(y,2)

        qc.h(qr[0])
        qc.cswap(qr[0], qr[1], qr[2])
        qc.h(qr[0])

        qc.measure(qr[0], cr[0])
        qc.reset(qr)

    #print('----before run----')
    job = execute(qc,backend=backend, shots=1024)
    #print('----after run----')
    result = job.result()
    data = result.get_counts()
    
    if len(data)==1:
        return 0.0
    else:
        return data['001']/1024.0

class QuantumKMeans():
    def __init__(self, backend, n_clusters=2, tol=0.0001, max_iter=300, verbose=False):
        self.K = n_clusters
        self.cluster_centers_ = np.empty(0)
        self.labels_ = np.empty(0)
        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.backend = backend
    
    def fit(self, X):
        finished = False
        self.cluster_centers_ = X.sample(n=self.K).reset_index(drop=True)
        iteration = 0
        while not finished and iteration<self.max_iter:
            if self.verbose: print("Iteration",iteration)
            distances = np.asarray([[distance(point,centroid,self.backend) for _,point in X.iterrows()] for _,centroid in self.cluster_centers_.iterrows()])
            self.labels_ = np.asarray([np.argmin(distances[:,i]) for i in range(distances.shape[1])])
            new_centroids = X.groupby(self.labels_).mean()
            if self.verbose: print("Old centroids are",self.cluster_centers_)
            if self.verbose: print("New centroids are",new_centroids)

            if abs((new_centroids - self.cluster_centers_).sum(axis=0).sum()) < self.tol:
                finished = True
            self.cluster_centers_ = new_centroids
            if self.verbose: print("Centers are", self.labels_)
            iteration += 1