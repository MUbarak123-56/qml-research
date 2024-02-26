import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("../data/train_small.csv")

cols = ['temperature', 'humidity', 'light', 'co2', 'humidityratio']

x_train = train[cols]
y_train = train["target"]

from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import BasicAer
from qiskit.utils import algorithm_globals
import time

num_qubits = len(cols)
algorithm_globals.random_seed = 12345

# Define the feature map
feature_map = PauliFeatureMap(feature_dimension=num_qubits, reps=1)

# Define the quantum kernel
qkernel = FidelityQuantumKernel(feature_map=feature_map)

# Initialize the QSVC model
model = QSVC(quantum_kernel=qkernel)


feat = pd.DataFrame()
for i in range(1,len(cols)):
    new_x = np.array(x_train.loc[:,cols[:i+1]])
    # Define the feature map
    feature_map = PauliFeatureMap(feature_dimension=new_x.shape[1], reps=1)

    # Define the quantum kernel
    qkernel = FidelityQuantumKernel(feature_map=feature_map)

    # Initialize the QSVC model
    model = QSVC(quantum_kernel=qkernel)
    start = time.time()
    model.fit(new_x, np.array(y_train))
    stop = time.time()
    elapsed=stop-start
    feat.loc[i, "num_features"] = i + 1
    feat.loc[i, "model"] = "QSVC"
    feat.loc[i, "runtime"] = elapsed
    #feat.loc[i,"kernel"] = typ
    print("feat ", i)
feat.to_csv("../results/runtime_features/qsvc.csv", index=False)