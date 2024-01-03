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

df_train=pd.read_csv("../data/train_fe_small.csv")
#df_test=pd.read_csv("../data/milk_test_fe.csv")
cols = ['region_South', 'region_West', 'account_length','number_vmail_messages', 'total_day_minutes', 'total_day_calls',
        'total_intl_charge', 'customer_service_calls', 'churn']
df_train = df_train[cols]

from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df_train, random_state = 42, train_size=0.8)

sampler = Sampler()
#optimizer = COBYLA(maxiter = 300)
#optimizer = SPSA(maxiter = 300)

X_train = df_train.drop(columns=['churn'])
y_train = df_train['churn']

X_val = df_val.drop(columns=['churn'])
y_val = df_val['churn']

x_train_arr = np.array(X_train)
x_val_arr = np.array(X_val)

y_train=y_train.to_numpy()
y_val = y_val.to_numpy()


from qiskit.circuit import ParameterVector, Parameter

### Feature map
pauli_feature_map = PauliFeatureMap(feature_dimension=len(X_train.columns),reps=1, paulis=['ZZ'])

### Ansatzes
ansatz_su = EfficientSU2(num_qubits=pauli_feature_map.width(), reps = 1, su2_gates=["ry", "rz"], entanglement= "full",
                         insert_barriers=True)
ansatz_two_local = TwoLocal(num_qubits=pauli_feature_map.width(),rotation_blocks=["ry", "rz"],entanglement_blocks="cx",
                                     entanglement="linear", reps=2, insert_barriers=True)

theta = Parameter("θ")
ansatz_n_local = NLocal(num_qubits=pauli_feature_map.width(),rotation_blocks=[RXGate(theta), CRZGate(theta)],
                        entanglement_blocks=CCXGate(),
                        entanglement=[[0, 1, 2], [0,2,1]],reps=2,insert_barriers=True)

### Optimizers
num_iter=300
cobyla = COBYLA(maxiter = num_iter)
spsa = SPSA(maxiter = num_iter)
reps = 1

def plot_confusion_matrix(conf_matrix, ansatz, optimizer):
    #num = len(os.listdir("../vqc_conf/train_"))
    conf_num = "../vqc_conf/train_process/" + str(ansatz) + "_" + str(optimizer) + "_" + str(reps) + ".png"
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(conf_num)
    plt.show()
    
def vqc_exp(ansatz, optimizer):
    
    if optimizer == "cobyla":
        optim_use = cobyla
    elif optimizer == "spsa":
        optim_use = spsa
        
    if ansatz == "su2":
        ansatz_use = ansatz_su
    elif ansatz == "two_local":
        ansatz_use = ansatz_two_local
    elif ansatz == "n_local":
        ansatz_use = ansatz_n_local
        
    #objective_func_vals = []
    objective_func_vals = []
    
    def callback_res(weights, obj_func_eval):
        clear_output(wait=True)
        objective_func_vals.append(obj_func_eval)
        print(objective_func_vals)
        return objective_func_vals
    
    print("===="*20)
    vqc = VQC(
        sampler=sampler,
        feature_map=pauli_feature_map,
        ansatz=ansatz_use,
        optimizer=optim_use,
        callback=callback_res,
    )

    start = time.time()
    vqc.fit(x_train_arr, y_train)
    elapsed = time.time() - start

    #num = len(os.listdir("../vqc_model/pauli"))
    vqc_mod_num = "../vqc_model/train_process/" + str(ansatz) + "_" + str(optimizer) + "_" + str(reps) + ".model"
    vqc.save(vqc_mod_num)

    print(f"Training time: {round(elapsed)} seconds")

    train_score = vqc.score(x_train_arr, y_train)
    val_score = vqc.score(x_val_arr, y_val)

    #vqc.save("vqc_enc_classifier.model")

    print(f"VQC on the training dataset: {train_score:.2f}")
    print(f"VQC on the val dataset:     {val_score:.2f}")
    
    pred_vqc = vqc.predict(x_val_arr)
    conf_matrix = confusion_matrix(y_val, pred_vqc)

    plot_confusion_matrix(conf_matrix, ansatz, optimizer)
    print(classification_report(y_val, pred_vqc))
    print("\n")

    df = pd.DataFrame({"one":[1]})

    df["feature_map_type"] = "Pauli"
    df["optimizer"] = optimizer
    df["train_score"] = train_score
    df["val_score"] = val_score
    df["recall_score"] = recall_score(y_val, pred_vqc)
    df["f1_score"] = f1_score(y_val, pred_vqc)
    df["precision_score"] = precision_score(y_val, pred_vqc)
    #df["Quantum Kernel"] = "No"
    #df["PCA"] = "No"
    df["objective_vals"] = str(objective_func_vals)
    df["Ansatz"] = ansatz
    df["Training time"] = elapsed
    df["Model"] = "VQC"
    df["Max Iter"] = num_iter
    df["reps"] = reps

    df = df.drop("one", axis = 1)

    #num = len(os.listdir("../vqc_results/pauli"))
    df.to_csv("../vqc_results/train_process/" + str(ansatz) + "_" + str(optimizer)  + "_" + str(reps) + ".csv", index = False)
    # clear objective value history
    #objective_func_vals = []
    #return df
    
optim = ["cobyla", "spsa"]
ansatz_list = ["su2", "two_local", "n_local"]
for i in range(len(optim)):
    for j in range(len(ansatz_list)):
        vqc_exp(ansatz_list[j], optim[i])