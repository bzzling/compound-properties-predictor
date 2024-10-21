import gzip
import csv
import torch
from rdkit import Chem
from torch_geometric.data import Data

def unzip_and_parse(file_path):
    with gzip.open(file_path, 'rt') as f:
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

def process_and_output(input_file, output_file, property_name):
    data = unzip_and_parse(input_file)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles', 'property', 'graph'])
        for row in data:
            smiles = row['canonicalsmiles']
            property_value = float(row[property_name])
            graph = smiles_to_graph(smiles, property_value)
            writer.writerow([smiles, property_value, graph])

if __name__ == "__main__":
    input_file = 'data/pubchem_data.gz'
    output_file = 'data/processed_data.csv'
    property_name = 'hbondacc'  # Replace with actual property name
    process_and_output(input_file, output_file, property_name)