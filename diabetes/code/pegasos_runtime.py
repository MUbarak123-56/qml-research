import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("../data/train_fe.csv")

cols = ['type', 'air_temperature_k', 'process_temperature_k','rotational_speed_rpm', 'torque_nm', 'tool_wear_min']

x_train = train[cols]
y_train = train["target"]

import numpy as np
from sklearn.model_selection import cross_val_score
from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import BasicAer
from qiskit.utils import algorithm_globals
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time

num_qubits = 7
algorithm_globals.random_seed = 12345

# Define the feature map
feature_map = PauliFeatureMap(feature_dimension=num_qubits, reps=1)

# Define the quantum kernel
qkernel = FidelityQuantumKernel(feature_map=feature_map)

# Initialize the QSVC model
model = PegasosQSVC(quantum_kernel=qkernel, C=100, num_steps=300)

sizes = np.linspace(0.1,1,10)
sizes = list(sizes)

size = pd.DataFrame()
for i in range(len(sizes)):
    numbers = np.random.randint(0,high=len(train), size=round(len(train)*sizes[i]))
    new_x = np.array(x_train.iloc[numbers,:].reset_index(drop=True))
    new_y = np.array(y_train.iloc[numbers])
    start = time.time()
    model.fit(new_x, new_y)
    stop = time.time()
    elapsed=stop-start
    size.loc[i, "size"] = sizes[i]*len(train)
    size.loc[i, "model"] = "Pegasos QSVC"
    size.loc[i, "runtime"] = elapsed
    #size.loc[i,"kernel"] = typ

size.to_csv("../results/runtime_size/pegasos_qsvc.csv", index=False)

feat = pd.DataFrame()
for i in range(1,len(cols)):
    new_x = np.array(x_train.loc[:,cols[:i+1]])
    # Define the feature map
    feature_map = PauliFeatureMap(feature_dimension=new_x.shape[1], reps=1)

    # Define the quantum kernel
    qkernel = FidelityQuantumKernel(feature_map=feature_map)

    # Initialize the QSVC model
    model = PegasosQSVC(quantum_kernel=qkernel, C=750, num_steps=300)
    start = time.time()
    model.fit(new_x, np.array(y_train))
    stop = time.time()
    elapsed=stop-start
    feat.loc[i, "num_features"] = i + 1
    feat.loc[i, "model"] = "Pegasos QSVC"
    feat.loc[i, "runtime"] = elapsed
    #feat.loc[i,"kernel"] = typ
feat.to_csv("../results/runtime_features/pegasos_qsvc.csv", index=False)