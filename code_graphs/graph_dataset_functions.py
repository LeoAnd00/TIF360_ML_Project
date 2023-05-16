from rdkit import Chem  # To extract information of the molecules
import numpy as np
import torch
from torch_geometric.data import Data


def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]  # If the atom is not in the list, get "Unknown"
        
    binary_encoding = [int(boolean) for boolean in list(map(lambda s: x==s, permitted_list))]
    
    return binary_encoding    


#### Atom featurisation ####
# Currently generates ca. 80 node features
def get_atom_features(atom, use_chirality = True, hydrogens_implicit = True):
    # list of permitted atoms
    permitted_atom_list = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca',
                           'Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co',
                           'Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt',
                           'Hg','Pb','Unknown']
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_atom_list)
    
    n_heavy_neighbors = one_hot_encoding(int(atom.GetDegree()), [0,1,2,3,4,"MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, 'Extreme'])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_ring_enc = one_hot_encoding(int(atom.IsInRing()), [0, 1])
    
    is_aromatic_enc = one_hot_encoding(int(atom.GetIsAromatic()), [0, 1])
    
    #Remove the numerical features, since they do not seem to add sufficient information
    # atomic_mass = [float(atom.GetMass())] # information contained in atom_type
    
    # vdw_radius = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())))] 
    
    # covalent_radius = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())]
                              
    atom_feature_vector = atom_type_enc + n_heavy_neighbors + formal_charge_enc + hybridisation_type_enc + is_in_ring_enc + is_aromatic_enc 
    
    if use_chirality:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
        
    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
        
    # atom_feature_vector += atomic_mass + vdw_radius + covalent_radius
        
    return np.array(atom_feature_vector) 


##### Bond Featurisation #####

# Currently generates ca. 10 edge features
def get_bond_features(bond, use_stereochemistry=True):
    permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                            Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    
    bond_type_enc = one_hot_encoding(str(bond.GetBondType()), permitted_bond_types)
    
    bond_is_conjugated_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conjugated_enc + bond_is_in_ring_enc
    
    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
        
    return np.array(bond_feature_vector)


# Define function to generate dataset of labeled Pytorch Geometric Graphs
def create_graph_dataset_from_smiles(x_smiles, y):
    ## Inputs:
    # x_smiles = [smiles_1, smiles_2, ...], smiles representation of molecules
    # y = [y_1, y_2, ...] list of numerical labels for each smiles string, here chemical properties
    
    ## Outputs:
    # dataset = [data_1, data_2, ...] list of torch_geometric.data.Data objects representing molecular graphs
    
    dataset = []
    
    for (smiles, y_val) in zip(x_smiles, y):
        # convert smiles to molecular object
        mol = Chem.MolFromSmiles(smiles)
        
        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds() # each bond is represented twice in the adjacency matrix
        n_node_features = len(get_atom_features(mol.GetAtomWithIdx(0)))
        if n_nodes > 1:
            n_edge_features = len(get_bond_features(mol.GetBondBetweenAtoms(0,1)))
        else:
            n_edge_features = 0  # for single atom molecules -> no edges
        
        # construct node feature matrix X 
        X = np.zeros((n_nodes, n_node_features))
        
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
        
        X = torch.tensor(X, dtype=torch.float)
        
        # construct edge index array E, shape = (2, n_edges)
        (rows, cols) = np.nonzero(Chem.GetAdjacencyMatrix(mol))
        torch_rows = torch.tensor(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.tensor(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)
        
        # construct edge feature matrix EF
        EF = np.zeros((n_edges, n_edge_features))       # Note: generates zero matrix if n_edges = n_edge_features = 0
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        EF = torch.tensor(EF, dtype=torch.float)
        
        # construct label/y tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype=torch.float)
        
        # construct torch_geometric.data.Data object and append to dataset
        dataset.append(Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor))
        
    return dataset
        