import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.disable(logging.CRITICAL)

import sys
import glob
import numpy as np
import torch
import pandas as pd
import deepchem as dc
import pubchempy as pcp
from openpom.feat.graph_featurizer import GraphFeaturizer
from openpom.models.mpnn_pom import MPNNPOMModel


# Function to hit PubChem API to convert give SMILES into molecule name
def get_name_from_smiles(smiles: str):
    try:
        compounds = pcp.get_compounds(smiles, 'smiles')
        if compounds:
            name = compounds[0].iupac_name or compounds[0].synonyms[0]
            return name
    except:
        pass
    return smiles

# SMILES input
smiles_str = input("Enter SMILES: ")

# Get the SMILES from PubChem API
print('Looking up molecule name...')
label = get_name_from_smiles(smiles_str)

# Import ODOR_LABELS
df = pd.read_csv('openpom-repo/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv')
ODOR_LABELS = df.columns[2:].tolist()

featurizer = GraphFeaturizer()
dataset = dc.data.NumpyDataset(X=featurizer.featurize([smiles_str]))

# Model setup
model = MPNNPOMModel(
    n_tasks=len(ODOR_LABELS),
    batch_size=1,
    node_out_feats=100,
    edge_hidden_feats=75,
    edge_out_feats=100,
    num_step_message_passing=5,
    mpnn_residual=True,
    message_aggregator_type='sum',
    readout_type='set2set',
    num_step_set2set=3,
    num_layer_set2set=2,
    ffn_hidden_list=[392, 392],
    ffn_embeddings=256,
    ffn_activation='relu',
    ffn_dropout_p=0.12,
    ffn_dropout_at_input_no_act=False,
    self_loop=False,
)

checkpoint_paths = sorted(glob.glob('openpom-repo/models/ensemble_models/experiments_*/checkpoint2.pt'))

all_predictions = []
for path in checkpoint_paths:
    checkpoint = torch.load(path, map_location='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    preds = model.predict(dataset)
    all_predictions.append(preds[0])

avg_predictions = np.mean(all_predictions, axis=0)
probs = torch.sigmoid(torch.tensor(avg_predictions)).numpy()

# Present smmell prediction
print(f"\nTop predicted odors for:\nNAME:   {label}\nSMILES: {smiles_str}\n")
scored = sorted(zip(ODOR_LABELS, probs), key=lambda x: x[1], reverse=True)
for label, score in scored[:10]:
    print(f"  {label}: {score:.1%}")
