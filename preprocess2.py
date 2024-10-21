import csv
import torch
from rdkit import Chem
from torch_geometric.data import Data

def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    return data

def smiles_to_graph(smiles, property_value):
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    
    num_atoms = len(atoms)
    atom_features = [[atom.GetAtomicNum()] for atom in atoms]
    edge_index = []
    
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
    
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index, y=torch.tensor([property_value], dtype=torch.float))

def process_and_output(input_file, output_file):
    data = read_csv(input_file)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles', 'solubility', 'graph'])
        for row in data:
            smiles = row['smiles']
            solubility = float(row['measured log solubility in mols per litre'])
            graph = smiles_to_graph(smiles, solubility)
            writer.writerow([smiles, solubility, graph])

if __name__ == "__main__":
    input_file = 'data/delaney-processed.csv'
    output_file = 'data/gnn_input_data.csv'
    process_and_output(input_file, output_file)