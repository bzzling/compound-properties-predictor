from rdkit import Chem

mol = Chem.MolFromSmiles('OCl(=O)(=O)=O')
atoms = mol.GetAtoms()
bonds = mol.GetBonds()

print(len(atoms))
print(len(bonds))