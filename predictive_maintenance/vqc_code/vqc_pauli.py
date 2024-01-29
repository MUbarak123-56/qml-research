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

train = pd.read_csv("../data/train_fe.csv")
test = pd.read_csv("../data/test_fe.csv")

cols = ['age','roomservice', 'spa', 'vrdeck', 'homeplanet_earth', 'homeplanet_europa', 'homeplanet_mars', 'transported']

train = train[cols]
x_train_use, y_train_use = train.drop("transported", axis = 1), train["transported"]

x_train_use = x_train_use.to_numpy()
y_train_use = y_train_use.to_numpy()

#df_test=pd.read_csv("../data/milk_test_fe.csv")
#cols = ['age','roomservice', 'spa', 'vrdeck', 'homeplanet_earth', 'homeplanet_europa', 'homeplanet_mars', 'transported']
#df_train = df_train[cols]

#from sklearn.model_selection import train_test_split
#df_train, df_val = train_test_split(df_train, random_state = 42, train_size=0.8)

sampler = Sampler()
#optimizer = COBYLA(maxiter = 300)
#optimizer = SPSA(maxiter = 300)

#X_train = df_train.drop(columns=['transported'])
#y_train = df_train['transported']

test = test[cols]

x_test, y_test = test.drop("transported", axis =1), test["transported"]

x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

#x_train_arr = np.array(X_train)
#x_val_arr = np.array(X_val)

#y_train=y_train.to_numpy()
#y_val = y_val.to_numpy()


from qiskit.circuit import ParameterVector, Parameter

### Feature map
pauli_feature_map = PauliFeatureMap(feature_dimension=x_train_use.shape[1],reps=1, paulis=['ZZ'])

reps = 3
### Ansatzes
ansatz_su = EfficientSU2(num_qubits=pauli_feature_map.width(), reps = reps, su2_gates=["ry", "rz"], entanglement= "full",
                         insert_barriers=True)
ansatz_two_local = TwoLocal(num_qubits=pauli_feature_map.width(),rotation_blocks=["ry", "rz"],entanglement_blocks="cx",
                                     entanglement="linear", reps=reps, insert_barriers=True)

theta = Parameter("θ")
ansatz_n_local = NLocal(num_qubits=pauli_feature_map.width(),rotation_blocks=[RXGate(theta), CRZGate(theta)],
                        entanglement_blocks=CCXGate(),
                        entanglement=[[0, 1, 2], [0,2,1]],reps=reps,insert_barriers=True)

### Optimizers
num_iter=100
cobyla = COBYLA(maxiter = num_iter)
spsa = SPSA(maxiter = num_iter)
#reps = 2

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
    vqc.fit(x_train_use, y_train_use)
    elapsed = time.time() - start

    #num = len(os.listdir("../vqc_model/pauli"))
    vqc_mod_num = "../vqc_model/train_process/" + str(ansatz) + "_" + str(optimizer) + "_" + str(reps) + ".model"
    vqc.save(vqc_mod_num)

    print(f"Training time: {round(elapsed)} seconds")

    #train_score = vqc.score(x_train_arr, y_train)
    #val_score = vqc.score(x_val_arr, y_val)

    #vqc.save("vqc_enc_classifier.model")

    #print(f"VQC on the training dataset: {train_score:.2f}")
    #print(f"VQC on the val dataset:     {val_score:.2f}")
    
    pred_train = vqc.predict(x_train_use)
    pred_test = vqc.predict(x_test)
    conf_matrix = confusion_matrix(y_test, pred_test)

    plot_confusion_matrix(conf_matrix, ansatz, optimizer)
    print(classification_report(y_test, pred_test))
    print("\n")

    df = pd.DataFrame({"one":[1]})

    f1_train = f1_score(y_train_use, pred_train)
    prec_train = precision_score(y_train_use, pred_train)
    recall_train = recall_score(y_train_use, pred_train)


    f1_test = f1_score(y_test, pred_test)
    prec_test = precision_score(y_test, pred_test)
    recall_test = recall_score(y_test, pred_test)

    df["f1_train"] = f1_train
    df["f1_test"] = f1_test
    df["prec_train"] = prec_train
    df["prec_test"] = prec_test
    df["recall_train"] = recall_train
    df["recall_test"] = recall_test
    df["elapsed"] = elapsed
    df["feature_map_type"] = "Pauli"
    df["optimizer"] = optimizer
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