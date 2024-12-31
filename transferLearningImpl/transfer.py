import pandas as pd
import numpy as np
from rdkit import Chem
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

# Load peptide data
filepath_raw = 'ToxinSequenceSMILES.xlsx'
data_file = pd.read_excel(filepath_raw, header=0, usecols=["SEQUENCE", "TOXICITY", "SMILES"])

# Initialize an empty list to store tuples
listOfTuples = []

# Iterate through each row to extract the SMILES and TOXICITY columns
for index, row in data_file.iterrows():
    molecule = (row["SMILES"], row["TOXICITY"])
    listOfTuples.append(molecule)

# Convert the data to StellarGraph objects
all_elements = []
lst_degree = []
lst_formal_charge = []
lst_radical_electrons = []
lst_hybridization = []
lst_aromatic = []

for molecule in listOfTuples:
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

graphs = []
labels = []

for molecule in listOfTuples:
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
        hybridization = (atom.GetHybridization().real - min_hybridization) / (max_hybridization - min_hybridization)
        aromatic = (atom.GetIsAromatic() - min_aromatic) / (max_aromatic - min_aromatic)
        element_onehot = [0] * len(all_elements)
        element_onehot[element_to_index[element]] = 1
        node_features.append(element_onehot)
    node_features = np.array(node_features)

    # Convert the edges to a pandas DataFrame
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    # Create a StellarGraph object from the molecular graph
    G = StellarGraph(nodes=node_features, edges=edges_df)
    graphs.append(G)
    labels.append(smileLabel)

# Convert labels to a pandas Series and print value counts
graph_labels = pd.Series(labels)
print(graph_labels.value_counts().to_frame())

# Create a PaddedGraphGenerator
generator = PaddedGraphGenerator(graphs=graphs)

# Load the pre-trained model
model = load_model('toxicityModel25small.h5')

# Method 1: Freeze GNN layers
def freeze_gnn_layers(model):
    for layer in model.layers[:-4]:  # Assuming the last 4 layers are the adaptive readout and output layers
        layer.trainable = False

# Method 2: Freeze adaptive readout layer
def freeze_adaptive_readout_layer(model):
    for layer in model.layers[-4:]:  # Assuming the last 4 layers are the adaptive readout and output layers
        layer.trainable = False

# Method 3: Freeze all layers and add a new output layer
def freeze_all_layers_and_add_output(model):
    for layer in model.layers:
        layer.trainable = False
    x = model.layers[-2].output  # Get the output of the second last layer
    x = Dense(128, activation="relu")(x)
    x = Dropout(rate=0.2)(x)
    predictions = Dense(units=1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=model.input, outputs=predictions)
    return model

# Choose the method
method = 1  # Change this to 2 or 3 for other methods

if method == 1:
    freeze_gnn_layers(model)
elif method == 2:
    freeze_adaptive_readout_layer(model)
elif method == 3:
    model = freeze_all_layers_and_add_output(model)

# Compile the model
model.compile(
    optimizer=Adam(lr=0.0001),
    loss=binary_crossentropy,
    metrics=["acc"]
)

# Prepare the data for training
X_train, X_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.2, random_state=42)

train_gen = generator.flow(X_train, targets=y_train, batch_size=32, symmetric_normalization=False)
test_gen = generator.flow(X_test, targets=y_test, batch_size=32, symmetric_normalization=False)

# Fine-tune the model
model.fit(train_gen, epochs=100, validation_data=test_gen, shuffle=True)

# Save the fine-tuned model
model.save('fine_tuned_toxicityModel25small.h5')