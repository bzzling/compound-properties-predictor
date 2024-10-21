from rdkit import Chem

mol = Chem.MolFromSmiles('OCl(=O)(=O)=O')
atoms = mol.GetAtoms()
bonds = mol.GetBonds()

print(len(atoms))
print(len(bonds))

create a new preprocess.py file that converts the data in data/data.csv into a format that can serve as a input for a graph neural network. use the SMILES column to convert to a graph format as well, and the rest of the properties in the data/data.csv file are graph level properties
