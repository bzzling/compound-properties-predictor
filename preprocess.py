import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
from torch_geometric.data import Data

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Convert molecule to graph
    atom_features = []
    edge_index = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())
    
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    
    x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

def preprocess_data(input_csv, output_file):
    df = pd.read_csv(input_csv)
    
    data_list = []
    for _, row in df.iterrows():
        smiles = row['smiles']
        graph = smiles_to_graph(smiles)
        
        # Add graph-level properties
        for col in df.columns:
            if col != 'smiles' and pd.api.types.is_numeric_dtype(df[col]):
                graph[col] = torch.tensor([row[col]], dtype=torch.float)
        
        data_list.append(graph)
    
    torch.save(data_list, output_file)

if __name__ == "__main__":
    preprocess_data('data/data.csv', 'data/processed_data.pt')