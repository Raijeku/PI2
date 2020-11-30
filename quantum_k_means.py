"""Module for quantum k-means algorithm with class containing sk-learn style functions resembling the k-means algorithm.

This module contains the QuantumKMeans class for clustering according to euclidian distances calculated by running quantum circuits. 

Typical usage example:

    import numpy as np
    import pandas as pd
    backend = Aer.get_backend('qasm_simulator')
    X = pd.DataFrame(np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]))
    quantum_k_means = QuantumKMeans(backend, n_clusters=2, verbose=True)
    quantum_k_means.fit(X)
    print(kmeans.labels_)
"""
import numpy as np
import pandas as pd
from qiskit import Aer, IBMQ, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from math import pi
from sklearn.preprocessing import normalize, scale

def preprocess(points,type='angle'):
    """Preprocesses data points according to a type criteria.

    The algorithm scales the data points if the type is 'angle' and normalizes the data points if the type is 'probability'.

    Args:

    points: The input data points.
    type: Preprocessing criteria for the data points.

    Returns:

    p_points: Preprocessed points.
    """
    if type == 'angle': p_points = scale(points)
    elif type == 'probability': p_points = normalize(points)
    return p_points

def distance(x,y,backend,type='angle'):
    """Calculates the distance between two data points using quantum circuits.

    The algorithm scales the data points if the type is 'angle' and normalizes the data points if the type is 'probability'.

    Args:

    x: The first data point.
    y: The second data point.
    backend: IBM quantum device to calculate the distance with.
    type: Routine for calculating distances, including data preprocessing and qubit preparation.

    Returns:

    distance: Distance between the two data points.
    """
    if type == 'angle':
        complexes_x = x[0] + 1j*x[1]
        complexes_y= y[0] + 1j*y[1]
        theta_1 = np.angle(complexes_x)
        theta_2 = np.angle(complexes_y)
    
        qr = QuantumRegister(3, name="qr")
        cr = ClassicalRegister(3, name="cr")

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
        qr = QuantumRegister(3, name="qr")
        cr = ClassicalRegister(3, name="cr")

        qc = QuantumCircuit(qr, cr, name="k_means")
        qc.initialize(x,1)
        qc.initialize(y,2)

        qc.h(qr[0])
        qc.cswap(qr[0], qr[1], qr[2])
        qc.h(qr[0])

        qc.measure(qr[0], cr[0])
        qc.reset(qr)

    job = execute(qc,backend=backend, shots=1024)
    result = job.result()
    data = result.get_counts()
    
    if len(data)==1:
        return 0.0
    else:
        if type == 'angle': return data['001']/1024.0
        elif type == 'probability': return data[list(data)[-1] == '1']

class QuantumKMeans():
    """Quantum k-means clustering algorithm. This k-means alternative implements Quantum Machine Learning to calculate distances between data points and centroids using quantum circuits.
    
    Args:
        n_clusters: The number of clusters to use and the amount of centroids generated.
        tol: Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
        verbose: Defines if verbosity is active for deeper insight into the class processes.
        max_iter: Maximum number of iterations of the quantum k-means algorithm for a single run.
        backend: IBM quantum device to run the quantum k-means algorithm on.
        type: {'angle', 'probability'} Specify the type of data encoding. 'angle': Uses U3 gates with its theta angle being the phase angle of the complex data point. 'probability': Relies on data normalization to preprocess the data to acquire a norm of 1.

    Attributes:
        cluster_centers_: Coordinates of cluster centers.
        labels_: Centroid labels for each data point.
        n_iter_: Number of iterations run before convergence.
    """
    def __init__(self, backend, n_clusters=2, tol=0.0001, max_iter=300, verbose=False, type='angle'):
        """Initializes an instance of the quantum k-means algorithm."""
        self.cluster_centers_ = np.empty(0)
        self.labels_ = np.empty(0)
        self.n_iter_ = 0
        self.n_clusters = n_clusters
        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.backend = backend
        self.type = type
    
    def fit(self, X):
        """Computes quantum k-means clustering.
        
        Args:
            X: Training instances to cluster.

        Returns:
            self: Fitted estimator.
        """
        finished = False
        X = pd.DataFrame(preprocess(X, self.type))
        self.cluster_centers_ = X.sample(n=self.n_clusters).reset_index(drop=True)
        iteration = 0
        while not finished and iteration<self.max_iter:
            if self.verbose: print("Iteration",iteration)
            distances = np.asarray([[distance(point,centroid,self.backend) for _,point in X.iterrows()] for _,centroid in self.cluster_centers_.iterrows()])
            self.labels_ = np.asarray([np.argmin(distances[:,i]) for i in range(distances.shape[1])])
            if self.type == 'angle': new_centroids = X.groupby(self.labels_).mean()
            elif self.type == 'probability': new_centroids = pd.DataFrame(preprocess(X.groupby(self.labels_).mean(),self.type))
            if self.verbose: print("Old centroids are",self.cluster_centers_)
            if self.verbose: print("New centroids are",new_centroids)

            if abs((new_centroids - self.cluster_centers_).sum(axis=0).sum()) < self.tol:
                finished = True
            self.cluster_centers_ = new_centroids
            if self.verbose: print("Centers are", self.labels_)
            self.n_iter_ += 1
        return self

    def predict(self, X, sample_weight = None):
        """Predict the closest cluster each sample in X belongs to.

        Args:
            X: New data points to predict.
            sample_weight: The weights for each observation in X. If None, all observations are assigned equal weight.

        Returns:
            labels: Centroid labels for each data point.
        """
        if sample_weight is None:
            distances = np.asarray([[distance(point,centroid,self.backend) for _,point in X.iterrows()] for _,centroid in self.cluster_centers_.iterrows()])
        else: 
            weight_X = X * sample_weight
            distances = np.asarray([[distance(point,centroid,self.backend) for _,point in weight_X.iterrows()] for _,centroid in self.cluster_centers_.iterrows()])
        labels = np.asarray([np.argmin(distances[:,i]) for i in range(distances.shape[1])])
        return labels