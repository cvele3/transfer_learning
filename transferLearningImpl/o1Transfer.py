import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

# Ako je potrebno za custom slojeve iz StellarGrapha:
from stellargraph.layer import DeepGraphCNN
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph import StellarGraph

from rdkit import Chem

# ======================================================================
# 1) UČITAVANJE PEPTIDA IZ XLSX-A
#    Pretpostavljamo da su kolone: SEQUENCE, TOXICITY, SMILES
# ======================================================================
filepath_peptides = "ToxinSequenceSMILES.xlsx"  # prilagodite naziv/putanju

data_peptides = pd.read_excel(filepath_peptides, header=0, usecols=["SEQUENCE", "TOXICITY", "SMILES"])

# Napravimo listu (smiles, label)
listOfTuples_peptides = []
for index, row in data_peptides.iterrows():
    smiles = row["SMILES"]
    label = row["TOXICITY"]
    listOfTuples_peptides.append((smiles, label))

print("Broj peptidnih zapisa:", len(listOfTuples_peptides))

# ======================================================================
# 2) KREIRANJE GRAFSKIH OBJEKATA (StellarGraph) - ISKLJUČIVO ONE-HOT
# ======================================================================
ZeroActivity = 0
OneActivity = 0
stellarGraphAllList = []

from collections import Counter

# Get a list of all unique elements in the molecules and extract features.
# At the same time, create a list of seen properties from each atom.
all_elements = []
lst_degree = []
lst_formal_charge = []
lst_radical_electrons = []
lst_hybridization = []
lst_aromatic = []
for molecule in listOfTuples_peptides:
    mol = Chem.MolFromSmiles(molecule[0])
    atoms = mol.GetAtoms()
    for atom in atoms:
        lst_degree.append(atom.GetDegree())
        lst_formal_charge.append(atom.GetFormalCharge())
        lst_radical_electrons.append(atom.GetNumRadicalElectrons())
        lst_hybridization.append(atom.GetHybridization().real)
        lst_aromatic.append(atom.GetIsAromatic())

        element = atom.GetSymbol()
        if element not in all_elements:
            all_elements.append(element)

# Determine min and max values for each property/feature.
min_degree, max_degree = min(lst_degree), max(lst_degree)
min_formal_charge, max_formal_charge = min(lst_formal_charge), max(lst_formal_charge)
min_radical_electrons, max_radical_electrons = min(lst_radical_electrons), max(lst_radical_electrons)
min_hybridization, max_hybridization = min(lst_hybridization), max(lst_hybridization)
min_aromatic, max_aromatic = min(lst_aromatic), max(lst_aromatic)

# Create a dictionary that maps each element to a unique index
element_to_index = {element: index for index, element in enumerate(all_elements)}

for molecule in listOfTuples_peptides:
    smileString = molecule[0]
    smileLabel = molecule[1]

    # Convert the SMILES string into a molecular graph using RDKit
    mol = Chem.MolFromSmiles(smileString)
    atoms = mol.GetAtoms()
    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))

    # Convert the RDKit atom objects to a Numpy array of node features
    node_features = []
    for atom in atoms:
        element = atom.GetSymbol()
        degree = (atom.GetDegree() - min_degree) / (max_degree - min_degree)
        # formal_charge = ...
        # num_radical_electrons = ...
        hybridization = (atom.GetHybridization().real - min_hybridization) / (max_hybridization - min_hybridization) if (max_hybridization - min_hybridization) != 0 else 0
        aromatic = (atom.GetIsAromatic() - min_aromatic) / (max_aromatic - min_aromatic) if (max_aromatic - min_aromatic) != 0 else 0

        element_onehot = [0] * len(all_elements)
        element_onehot[element_to_index[element]] = 1

        # isključivo one-hot:
        node_features.append(element_onehot)
        # ako želite i (degree, hybridization, aromatic), otkomentirajte donji red:
        # node_features.append(element_onehot + [degree, hybridization, aromatic])

    node_features = np.array(node_features)
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    # Create a StellarGraph object from the molecular graph
    G = StellarGraph(nodes=node_features, edges=edges_df)

    if smileLabel == 1:
        OneActivity += 1
        stellarGraphAllList.append((G, smileLabel))
    if smileLabel == 0:
        ZeroActivity += 1
        stellarGraphAllList.append((G, smileLabel))

print("ZeroActivity:", ZeroActivity)
print("OneActivity:", OneActivity)
print("Ukupno grafo-va:", len(stellarGraphAllList))

import pandas as pd

graphs = []
labels = []

for triple in stellarGraphAllList:
    grafinjo = triple[0]
    active = triple[1]
    graphs.append(grafinjo)
    labels.append(active)

graph_labels = pd.Series(labels)
print("Distribucija labela:\n", graph_labels.value_counts().to_frame())

generator = PaddedGraphGenerator(graphs=graphs)

# ======================================================================
# 3) PRIPREMA train/val SPLIT (da dobijemo train_gen_pep i val_gen_pep)
# ======================================================================
graphs_pep = np.array(graphs)
labels_pep = np.array(labels)

X_train_pep, X_val_pep, y_train_pep, y_val_pep = train_test_split(
    graphs_pep, labels_pep, test_size=0.2, random_state=42, stratify=labels_pep
)

pep_generator = PaddedGraphGenerator(graphs=graphs_pep)

train_gen_pep = pep_generator.flow(X_train_pep, y_train_pep, batch_size=32, shuffle=True)
val_gen_pep   = pep_generator.flow(X_val_pep,   y_val_pep,   batch_size=32, shuffle=False)


# ======================================================================
# 4) UČITAVANJE POSTOJEĆEG, VEĆ ISTRENIRANOG MODELA + TRANSFER LEARNING
# ======================================================================
PRETRAINED_MODEL_PATH = "toxicityModel25small.h5"

def load_pretrained_model():
    # Ako vaš .h5 koristi DeepGraphCNN, trebate:
    model = load_model(
        PRETRAINED_MODEL_PATH,
        custom_objects={"DeepGraphCNN": DeepGraphCNN}
    )
    return model

# -------------------------------------------------------------------------
# 5) METODA 1: ZAMRZAVANJE GNN SLOJEVA
# -------------------------------------------------------------------------
print("\n=== METODA 1: Zamrzavanje GNN slojeva ===")
model1 = load_pretrained_model()

# Pretpostavimo da su GNN slojevi imenovani "deep_graph_cnn", "graph_conv", ...
for layer in model1.layers:
    if "deep_graph_cnn" in layer.name or "graph_conv" in layer.name:
        layer.trainable = False
    else:
        layer.trainable = True

model1.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=binary_crossentropy,
    metrics=["accuracy"]
)

history1 = model1.fit(
    train_gen_pep,
    validation_data=val_gen_pep,
    epochs=5,  # DEMO
    verbose=1
)
model1.save("toxicityModel25small_peptidi_freezeGNN.h5")

# -------------------------------------------------------------------------
# 6) METODA 2: ZAMRZAVANJE READOUT/DENSE SLOJEVA, TRENIRANJE GNN
# -------------------------------------------------------------------------
print("\n=== METODA 2: Zamrzavanje READOUT sloja ===")

model2 = load_pretrained_model()

for layer in model2.layers:
    if any(x in layer.name for x in ["dense", "dropout", "flatten", "readout"]):
        layer.trainable = False
    else:
        layer.trainable = True

model2.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=binary_crossentropy,
    metrics=["accuracy"]
)

history2 = model2.fit(
    train_gen_pep,
    validation_data=val_gen_pep,
    epochs=5,  # DEMO
    verbose=1
)
model2.save("toxicityModel25small_peptidi_freezeReadout.h5")

# -------------------------------------------------------------------------
# 7) METODA 3: ZAMRZAVANJE SVIH SLOJEVA + DODAVANJE NOVOG IZLAZNOG SLOJA
# -------------------------------------------------------------------------
print("\n=== METODA 3: Zamrzavanje SVIH slojeva + novi izlazni sloj ===")

base_model = load_pretrained_model()

for layer in base_model.layers:
    layer.trainable = False

# Pretpostavimo da je stari izlaz *zadnji* sloj, pa uzmemo npr. pretposljednji:
intermediate_output = base_model.layers[-2].output
new_output = Dense(1, activation="sigmoid", name="new_output")(intermediate_output)

model3 = Model(inputs=base_model.input, outputs=new_output)

model3.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=binary_crossentropy,
    metrics=["accuracy"]
)

history3 = model3.fit(
    train_gen_pep,
    validation_data=val_gen_pep,
    epochs=5,  # DEMO
    verbose=1
)
model3.save("toxicityModel25small_peptidi_freezeAllNewOutput.h5")

# -------------------------------------------------------------------------
# KRAJ
# -------------------------------------------------------------------------
print("\nSva tri modela (one-hot atomske značajke) istrenirana i spremljena s različitim pristupima transfer learningu.")
