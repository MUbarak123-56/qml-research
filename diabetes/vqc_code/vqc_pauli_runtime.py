import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from qiskit import *
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import EfficientSU2, TwoLocal, NLocal, RealAmplitudes, ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qiskit.circuit.library import CCXGate, CRZGate, RXGate
from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA
from matplotlib import pyplot as plt
from IPython.display import clear_output
from qiskit.primitives import Sampler
from qiskit.circuit import ParameterVector
from sklearn.metrics import recall_score, precision_score, f1_score
import time
from qiskit_machine_learning.algorithms.classifiers import VQC
import os


train = pd.read_csv("../data/train_small.csv")

cols = ['pregnancy_occurence', 'glucose_concentration', 'blood_pressure',
       'triceps_foldness', 'serum_insulin', 'bmi', 'predigree_function', 'age']

x_train = train[cols]
y_train = train["target"]

sizes = np.linspace(0.1,1,8)
sizes = list(sizes)

num_iter=100
sampler = Sampler()
cobyla = COBYLA(maxiter = num_iter)
spsa = SPSA(maxiter = num_iter)

def vqc_runtime(ansatz, optimizer):
    size = pd.DataFrame()
    for i in range(len(sizes)):
        numbers = np.random.randint(0,high=len(train), size=round(len(train)*sizes[i]))
        new_x = x_train.iloc[numbers,:].reset_index(drop=True)
        new_y = np.array(y_train.iloc[numbers].reset_index(drop=True))
        pauli_feature_map = PauliFeatureMap(feature_dimension=len(new_x.columns), reps=1)
        if optimizer == "cobyla":
            optim_use = cobyla
        elif optimizer == "spsa":
            optim_use = spsa

        if ansatz == "su2":
            ansatz_use = EfficientSU2(num_qubits=pauli_feature_map.width(), reps = 1, su2_gates=["ry", "rz"],
                                      entanglement= "full", insert_barriers=True)
        elif ansatz == "two_local":
            ansatz_use = ansatz_two_local = TwoLocal(num_qubits=pauli_feature_map.width(),rotation_blocks=["ry", "rz"],
                                                     entanglement_blocks="cx", entanglement="linear", reps=1,
                                                     insert_barriers=True)
        elif ansatz == "n_local":
            #ansatz_use = ansatz_n_local
            theta = Parameter("θ")
            ansatz_use = NLocal(num_qubits=pauli_feature_map.width(),rotation_blocks=[RXGate(theta), CRZGate(theta)],
                        entanglement_blocks=CCXGate(),
                        entanglement=[[0, 1, 2], [0,2,1]],reps=1,insert_barriers=True)


        #ansatz = ansatz_use
        model = VQC(
            sampler=sampler,
            feature_map=pauli_feature_map,
            ansatz=ansatz_use,
            optimizer=optim_use)
        new_x_arr = new_x.to_numpy()
        #new_y_arr = new_y.to_numpy()
        start = time.time()
        model.fit(new_x_arr, new_y)
        stop = time.time()
        elapsed=stop-start
        size.loc[i, "size"] = sizes[i]*len(train)
        size.loc[i, "model"] = "vqc" + "_" + str(ansatz) + "_" + str(optimizer)
        size.loc[i, "runtime"] = elapsed
        #size.loc[i,"kernel"] = typ
        print("size: ", i)
    size.to_csv("../vqc_results/runtime_size/vqc" + "_" + str(ansatz) + "_" + str(optimizer) + ".csv", index=False)
    
    feat = pd.DataFrame()
    for i in range(1,len(cols)):
        new_x = x_train.loc[:,cols[:i+1]]
        new_x_arr = new_x.to_numpy()
        new_y = np.array(y_train)
        pauli_feature_map = PauliFeatureMap(feature_dimension=len(new_x.columns), reps=1)
        if optimizer == "cobyla":
            optim_use = cobyla
        elif optimizer == "spsa":
            optim_use = spsa

        if ansatz == "su2":
            ansatz_use = EfficientSU2(num_qubits=pauli_feature_map.width(), reps = 1, su2_gates=["ry", "rz"], entanglement= "full",
                             insert_barriers=True)
        elif ansatz == "two_local":
            ansatz_use = ansatz_two_local = TwoLocal(num_qubits=pauli_feature_map.width(),rotation_blocks=["ry", "rz"],
                                                     entanglement_blocks="cx", entanglement="linear", reps=1,
                                                     insert_barriers=True)
        elif ansatz == "n_local":
            #ansatz_use = ansatz_n_local
            theta = Parameter("θ")
            ansatz_use = NLocal(num_qubits=pauli_feature_map.width(),rotation_blocks=[RXGate(theta), CRZGate(theta)],
                        entanglement_blocks=CCXGate(),
                        entanglement=[[0, 1, 2], [0,2,1]],reps=1,insert_barriers=True)


        #ansatz = ansatz_use
        model = VQC(
            sampler=sampler,
            feature_map=pauli_feature_map,
            ansatz=ansatz_use,
            optimizer=optim_use)
        start = time.time()
        model.fit(new_x_arr, new_y)
        stop = time.time()
        elapsed=stop-start
        feat.loc[i, "num_features"] = i + 1
        feat.loc[i, "model"] = "vqc" + "_" + str(ansatz) + "_" + str(optimizer)
        feat.loc[i, "runtime"] = elapsed
        #feat.loc[i,"kernel"] = typ
        print("num of features: ", i)
    feat.to_csv("../vqc_results/runtime_features/vqc" + "_" + str(ansatz) + "_" + str(optimizer) + ".csv", index=False)

#optim = ["cobyla", "spsa"]
#ansatz_list = ["su2", "two_local", "n_local"]
optim = ["cobyla"]
ansatz_list = ["su2"]
for i in range(len(optim)):
    for j in range(len(ansatz_list)):
        vqc_runtime(ansatz_list[j], optim[i])