import numpy as np
from qubox.qap import QAP

factory_matrix = np.array([[0, 5, 2], [5, 0, 3], [2, 3, 0]])
distance_matrix = np.array([[0, 8, 15], [8, 0, 13], [15, 13, 0]])
ALPHA = 100

instance = QAP(
    factory_matrix=factory_matrix,
    distance_matrix=distance_matrix,
    ALPHA=ALPHA
    )