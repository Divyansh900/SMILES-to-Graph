import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumHBD, CalcNumHBA
from typing import Dict, List, Optional, Tuple
import warnings

class SMILESToGraph:
    """
    Enhanced SMILES to graph converter with configurable feature levels.
    
    Feature Levels:
    - basic: Essential structural features
    - standard: Basic + common chemical properties
    - extended: Standard + advanced descriptors
    - comprehensive: All available features
    """
    
    def __init__(self, 
                 feature_level: str = "standard",
                 include_3d: bool = False,
                 include_partial_charges: bool = False,
                 include_descriptors: bool = True,
                 max_atomic_num: int = 100,
                 common_atoms: List[str] = None):
        """
        Initialize the converter with specified feature configuration.
        
        Args:
            feature_level: One of ['basic', 'standard', 'extended', 'comprehensive']
            include_3d: Include 3D coordinates if available
            include_partial_charges: Compute and include Gasteiger partial charges
            include_descriptors: Include molecular descriptors
            max_atomic_num: Maximum atomic number for one-hot encoding
            common_atoms: List of atoms to one-hot encode (default: C, N, O, S, P, F, Cl, Br, I)
        """
        self.feature_level = feature_level.lower()
        self.include_3d = include_3d
        self.include_partial_charges = include_partial_charges
        self.include_descriptors = include_descriptors
        self.max_atomic_num = max_atomic_num
        
        if common_atoms is None:
            self.common_atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        else:
            self.common_atoms = common_atoms
            
        # Validate feature level
        valid_levels = ['basic', 'standard', 'extended', 'comprehensive']
        if self.feature_level not in valid_levels:
            raise ValueError(f"feature_level must be one of {valid_levels}")
    
    def _get_atom_features(self, atom, mol, ring_info=None, partial_charges=None) -> List[float]:
        """Extract atom features based on configuration."""
        features = []
        
        # Basic features (always included)
        atomic_num = atom.GetAtomicNum()
        features.extend([
            atomic_num,
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
        ])
        
        if self.feature_level in ['standard', 'extended', 'comprehensive']:
            # Standard additional features
            features.extend([
                atom.GetTotalValence() - atom.GetFormalCharge(),  # Implicit valence (approximate)
                atom.GetTotalValence(),  # Total valence (explicit + implicit)
                int(atom.IsInRing()),
                atom.GetMass(),
                int(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
                atom.GetTotalDegree(),
            ])
            
            # One-hot encoding for common atoms
            for atom_symbol in self.common_atoms:
                features.append(int(atom.GetSymbol() == atom_symbol))
            
            # Atomic number one-hot (for less common atoms)
            if atomic_num <= self.max_atomic_num:
                onehot = [0] * self.max_atomic_num
                onehot[atomic_num - 1] = 1
                features.extend(onehot[:20])  # Limit to first 20 elements to save space
        
        if self.feature_level in ['extended', 'comprehensive']:
            # Extended features
            atom_idx = atom.GetIdx()
            
            # Ring information
            if ring_info:
                ring_sizes = []
                for ring in ring_info.AtomRings():
                    if atom_idx in ring:
                        ring_sizes.append(len(ring))
                
                # Ring size features
                features.extend([
                    int(3 in ring_sizes),  # 3-membered ring
                    int(4 in ring_sizes),  # 4-membered ring
                    int(5 in ring_sizes),  # 5-membered ring
                    int(6 in ring_sizes),  # 6-membered ring
                    int(7 in ring_sizes),  # 7-membered ring
                    int(any(size > 7 for size in ring_sizes)),  # larger rings
                    len(ring_sizes),  # number of rings this atom is in
                ])
            
            # Additional chemical properties
            features.extend([
                int(atom.GetIsAromatic()),
                atom.GetTotalValence(),
                atom.GetNumRadicalElectrons(),
            ])
        
        if self.feature_level == 'comprehensive':
            # Comprehensive features
            features.extend([
                atom.GetAtomMapNum(),
                int(atom.HasProp('_ChiralityPossible')),
                atom.GetUnsignedProp('_CIPCode') if atom.HasProp('_CIPCode') else 0,
            ])
        
        # Optional features
        if self.include_partial_charges and partial_charges is not None:
            features.append(partial_charges[atom.GetIdx()])
        
        if self.include_3d and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            pos = conf.GetAtomPosition(atom.GetIdx())
            features.extend([pos.x, pos.y, pos.z])
        
        return features
    
    def _get_bond_features(self, bond) -> List[float]:
        """Extract bond features based on configuration."""
        bond_type = bond.GetBondType()
        features = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
        ]
        
        if self.feature_level in ['standard', 'extended', 'comprehensive']:
            features.extend([
                int(bond.IsInRing()),
                int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE),
            ])
        
        if self.feature_level in ['extended', 'comprehensive']:
            features.extend([
                1.0,  # Placeholder for valence contribution (bond-specific)
                1.0,  # Placeholder for valence contribution (bond-specific)
            ])
        
        if self.feature_level == 'comprehensive':
            features.extend([
                int(bond.GetBondDir() != Chem.rdchem.BondDir.NONE),
                bond.GetBondTypeAsDouble(),
            ])
        
        return features
    
    def _get_molecular_descriptors(self, mol) -> Dict[str, float]:
        """Calculate molecular descriptors."""
        if not self.include_descriptors:
            return {}
        
        descriptors = {}
        
        try:
            # Basic descriptors
            descriptors.update({
                'mol_weight': Descriptors.MolWt(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': rdMolDescriptors.CalcNumRings(mol),
            })
            
            if self.feature_level in ['standard', 'extended', 'comprehensive']:
                descriptors.update({
                    'logp': Descriptors.MolLogP(mol),
                    'tpsa': CalcTPSA(mol),
                    'hbd': CalcNumHBD(mol),
                    'hba': CalcNumHBA(mol),
                    'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                    'num_saturated_rings': rdMolDescriptors.CalcNumSaturatedRings(mol),
                })
            
            if self.feature_level in ['extended', 'comprehensive']:
                try:
                    # Try BertzCT, fall back to alternative if not available
                    if hasattr(rdMolDescriptors, 'BertzCT'):
                        bertz_ct = rdMolDescriptors.BertzCT(mol)
                    else:
                        # Alternative complexity measure using number of bonds and rings
                        bertz_ct = mol.GetNumBonds() + rdMolDescriptors.CalcNumRings(mol) * 2
                    
                    descriptors.update({
                        'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
                        'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                        'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
                        'bertz_ct': bertz_ct,
                        'balaban_j': rdMolDescriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0,
                    })
                except Exception as e:
                    # Minimal descriptors if advanced ones fail
                    descriptors.update({
                        'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
                        'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                        'bertz_ct': mol.GetNumBonds() + rdMolDescriptors.CalcNumRings(mol) * 2,
                        'balaban_j': 0,
                    })
                    if hasattr(rdMolDescriptors, 'CalcFractionCSP3'):
                        descriptors['fraction_csp3'] = rdMolDescriptors.CalcFractionCSP3(mol)
            
            if self.feature_level == 'comprehensive':
                try:
                    # Check for availability of each descriptor individually
                    comprehensive_descriptors = {}
                    
                    # Kappa indices
                    if hasattr(rdMolDescriptors, 'Kappa1'):
                        comprehensive_descriptors['kappa1'] = rdMolDescriptors.Kappa1(mol)
                    else:
                        comprehensive_descriptors['kappa1'] = mol.GetNumAtoms()
                    
                    if hasattr(rdMolDescriptors, 'Kappa2'):
                        comprehensive_descriptors['kappa2'] = rdMolDescriptors.Kappa2(mol)
                    else:
                        comprehensive_descriptors['kappa2'] = mol.GetNumBonds()
                    
                    if hasattr(rdMolDescriptors, 'Kappa3'):
                        comprehensive_descriptors['kappa3'] = rdMolDescriptors.Kappa3(mol)
                    else:
                        comprehensive_descriptors['kappa3'] = rdMolDescriptors.CalcNumRings(mol)
                    
                    # Chi indices
                    if hasattr(rdMolDescriptors, 'Chi0v'):
                        comprehensive_descriptors['chi0v'] = rdMolDescriptors.Chi0v(mol)
                    else:
                        comprehensive_descriptors['chi0v'] = mol.GetNumAtoms()
                    
                    if hasattr(rdMolDescriptors, 'Chi1v'):
                        comprehensive_descriptors['chi1v'] = rdMolDescriptors.Chi1v(mol)
                    else:
                        comprehensive_descriptors['chi1v'] = mol.GetNumBonds()
                    
                    if hasattr(rdMolDescriptors, 'Chi2v'):
                        comprehensive_descriptors['chi2v'] = rdMolDescriptors.Chi2v(mol)
                    else:
                        comprehensive_descriptors['chi2v'] = rdMolDescriptors.CalcNumRings(mol)
                    
                    # Hall-Kier alpha
                    if hasattr(rdMolDescriptors, 'HallKierAlpha'):
                        comprehensive_descriptors['hall_kier_alpha'] = rdMolDescriptors.HallKierAlpha(mol)
                    else:
                        # Simple flexibility measure: rotatable_bonds / total_bonds
                        comprehensive_descriptors['hall_kier_alpha'] = (
                            rdMolDescriptors.CalcNumRotatableBonds(mol) / max(mol.GetNumBonds(), 1)
                        )
                    
                    descriptors.update(comprehensive_descriptors)
                    
                except Exception as e:
                    warnings.warn(f"Error in comprehensive descriptors: {e}")
                    # Minimal fallback
                    descriptors.update({
                        'kappa1': mol.GetNumAtoms(),
                        'kappa2': mol.GetNumBonds(), 
                        'kappa3': rdMolDescriptors.CalcNumRings(mol),
                        'chi0v': mol.GetNumAtoms(),
                        'chi1v': mol.GetNumBonds(),
                        'chi2v': rdMolDescriptors.CalcNumRings(mol), 
                        'hall_kier_alpha': rdMolDescriptors.CalcNumRotatableBonds(mol) / max(mol.GetNumBonds(), 1),
                    })
                
        except Exception as e:
            warnings.warn(f"Error calculating some descriptors: {e}")
        
        return descriptors
    
    def to_graph(self, smiles: str) -> Optional[Dict]:
        """
        Convert SMILES to graph representation.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary containing:
            - node_features: numpy array of atom features
            - edges: numpy array of edge indices
            - edge_features: numpy array of edge features
            - descriptors: dictionary of molecular descriptors
            - metadata: dictionary with feature information
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens if requested for 3D features
        if self.include_3d:
            mol = Chem.AddHs(mol)
        
        # Get ring information
        ring_info = mol.GetRingInfo()
        
        # Calculate partial charges if requested
        partial_charges = None
        if self.include_partial_charges:
            try:
                ComputeGasteigerCharges(mol)
                partial_charges = [float(atom.GetProp('_GasteigerCharge')) 
                                 for atom in mol.GetAtoms()]
            except:
                warnings.warn("Could not compute Gasteiger charges")
                partial_charges = [0.0] * mol.GetNumAtoms()
        
        # Extract atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = self._get_atom_features(atom, mol, ring_info, partial_charges)
            atom_features.append(features)
        
        # Extract bond features and build edge list
        edges = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edges.extend([[i, j], [j, i]])
            
            bond_feat = self._get_bond_features(bond)
            edge_features.extend([bond_feat, bond_feat])
        
        # Get molecular descriptors
        descriptors = self._get_molecular_descriptors(mol)
        
        # Metadata
        metadata = {
            'feature_level': self.feature_level,
            'num_atom_features': len(atom_features[0]) if atom_features else 0,
            'num_edge_features': len(edge_features[0]) if edge_features else 0,
            'include_3d': self.include_3d,
            'include_partial_charges': self.include_partial_charges,
            'smiles': smiles,
        }
        
        return {
            'node_features': np.array(atom_features, dtype=np.float32),
            'edges': np.array(edges, dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32),
            'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else np.empty((0, 0), dtype=np.float32),
            'descriptors': descriptors,
            'metadata': metadata
        }
    
    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """
        Get names of atom and bond features for the current configuration.
        
        Returns:
            Tuple of (atom_feature_names, bond_feature_names)
        """
        atom_names = ['atomic_num', 'formal_charge', 'hybridization', 'is_aromatic', 'degree', 'total_h']
        bond_names = ['single', 'double', 'triple', 'aromatic', 'conjugated']
        
        if self.feature_level in ['standard', 'extended', 'comprehensive']:
            atom_names.extend([
                'implicit_valence', 'explicit_valence', 'is_in_ring', 'mass', 
                'is_chiral', 'total_degree'
            ])
            atom_names.extend([f'is_{atom}' for atom in self.common_atoms])
            atom_names.extend([f'atomic_num_onehot_{i}' for i in range(1, 21)])
            
            bond_names.extend(['is_in_ring', 'has_stereo'])
        
        if self.feature_level in ['extended', 'comprehensive']:
            atom_names.extend([
                'in_3ring', 'in_4ring', 'in_5ring', 'in_6ring', 'in_7ring', 
                'in_large_ring', 'num_rings', 'total_valence', 'num_radical_electrons'
            ])
            bond_names.extend(['valence_contrib_begin', 'valence_contrib_end'])
        
        if self.feature_level == 'comprehensive':
            atom_names.extend(['atom_map_num', 'chirality_possible', 'cip_code'])
            bond_names.extend(['has_direction', 'bond_type_double'])
        
        if self.include_partial_charges:
            atom_names.append('partial_charge')
        
        if self.include_3d:
            atom_names.extend(['x', 'y', 'z'])
        
        return atom_names, bond_names

# Example usage and convenience functions
def create_converter(feature_level: str = "standard", **kwargs) -> SMILESToGraph:
    """Create a preconfigured converter."""
    return SMILESToGraph(feature_level=feature_level, **kwargs)

def quick_convert(smiles: str, feature_level: str = "standard") -> Optional[Dict]:
    """Quick conversion with default settings."""
    converter = SMILESToGraph(feature_level=feature_level)
    return converter.to_graph(smiles)

# Example configurations
PRESET_CONFIGS = {
    'drug_discovery': {
        'feature_level': 'extended',
        'include_partial_charges': True,
        'include_descriptors': True,
        'common_atoms': ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br']
    },
    'reaction_prediction': {
        'feature_level': 'comprehensive',
        'include_partial_charges': True,
        'include_descriptors': False,
        'common_atoms': ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    },
    'property_prediction': {
        'feature_level': 'standard',
        'include_descriptors': True,
        'include_partial_charges': False,
    },
    'minimal': {
        'feature_level': 'basic',
        'include_descriptors': False,
        'include_partial_charges': False,
    }
}
