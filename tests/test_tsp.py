import numpy as np

from qubingx.tsp import TSP

distance_matrix = np.array([[0, 8, 15], [8, 0, 13], [15, 13, 0]])
ALPHA = 100

instance = TSP(distance_matrix=distance_matrix, ALPHA=ALPHA)
