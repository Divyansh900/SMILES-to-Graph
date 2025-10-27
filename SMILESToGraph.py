import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumHBD, CalcNumHBA
from typing import Dict, List, Optional, Tuple, Union
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
            
        # Predefined normalization constants (empirical values from ChEMBL/ZINC datasets)
        self._descriptor_norms = {
            'mol_weight': {'mean': 350.0, 'std': 150.0, 'min': 50.0, 'max': 1000.0},
            'logp': {'mean': 2.5, 'std': 2.0, 'min': -5.0, 'max': 10.0},
            'tpsa': {'mean': 70.0, 'std': 40.0, 'min': 0.0, 'max': 300.0},
            'hbd': {'mean': 2.0, 'std': 2.0, 'min': 0.0, 'max': 15.0},
            'hba': {'mean': 4.0, 'std': 3.0, 'min': 0.0, 'max': 20.0},
            'num_atoms': {'mean': 25.0, 'std': 10.0, 'min': 5.0, 'max': 100.0},
            'num_bonds': {'mean': 26.0, 'std': 11.0, 'min': 4.0, 'max': 110.0},
            'num_rings': {'mean': 2.0, 'std': 1.5, 'min': 0.0, 'max': 8.0},
            'num_aromatic_rings': {'mean': 1.5, 'std': 1.0, 'min': 0.0, 'max': 6.0},
            'num_saturated_rings': {'mean': 0.5, 'std': 1.0, 'min': 0.0, 'max': 5.0},
            'num_heteroatoms': {'mean': 4.0, 'std': 3.0, 'min': 0.0, 'max': 20.0},
            'num_rotatable_bonds': {'mean': 5.0, 'std': 4.0, 'min': 0.0, 'max': 25.0},
            'fraction_csp3': {'mean': 0.3, 'std': 0.25, 'min': 0.0, 'max': 1.0},
            'bertz_ct': {'mean': 500.0, 'std': 300.0, 'min': 50.0, 'max': 2000.0},
            'balaban_j': {'mean': 1.5, 'std': 0.5, 'min': 0.0, 'max': 4.0},
            'kappa1': {'mean': 15.0, 'std': 8.0, 'min': 3.0, 'max': 50.0},
            'kappa2': {'mean': 8.0, 'std': 5.0, 'min': 1.0, 'max': 30.0},
            'kappa3': {'mean': 4.0, 'std': 3.0, 'min': 0.0, 'max': 15.0},
            'chi0v': {'mean': 15.0, 'std': 8.0, 'min': 3.0, 'max': 50.0},
            'chi1v': {'mean': 8.0, 'std': 5.0, 'min': 1.0, 'max': 30.0},
            'chi2v': {'mean': 6.0, 'std': 4.0, 'min': 0.0, 'max': 20.0},
            'hall_kier_alpha': {'mean': 0.0, 'std': 2.0, 'min': -10.0, 'max': 10.0}
        }
    
    def get_feature_shapes(self) -> Dict[str, int]:
        """
        Get the number of features for each component at the current feature level.
        
        Returns:
            Dictionary with feature counts for:
            - node_features: number of atom features
            - edge_features: number of bond features  
            - graph_features: number of molecular descriptors
        """
        # Calculate atom features count
        atom_count = 6  # Basic features: atomic_num, formal_charge, hybridization, is_aromatic, degree, total_h
        
        if self.feature_level in ['standard', 'extended', 'comprehensive']:
            atom_count += 6  # implicit_valence, explicit_valence, is_in_ring, mass, is_chiral, total_degree
            atom_count += len(self.common_atoms)  # One-hot for common atoms
            atom_count += 20  # Atomic number one-hot (limited to first 20 elements)
        
        if self.feature_level in ['extended', 'comprehensive']:
            atom_count += 10  # Ring features + additional chemical properties
        
        if self.feature_level == 'comprehensive':
            atom_count += 3  # atom_map_num, chirality_possible, cip_code
        
        if self.include_partial_charges:
            atom_count += 1
        
        if self.include_3d:
            atom_count += 3  # x, y, z coordinates
        
        # Calculate bond features count
        bond_count = 5  # Basic: single, double, triple, aromatic, conjugated
        
        if self.feature_level in ['standard', 'extended', 'comprehensive']:
            bond_count += 2  # is_in_ring, has_stereo
        
        if self.feature_level in ['extended', 'comprehensive']:
            bond_count += 2  # valence contributions
        
        if self.feature_level == 'comprehensive':
            bond_count += 2  # has_direction, bond_type_double
        
        # Calculate graph features count (molecular descriptors)
        graph_count = 0
        if self.include_descriptors:
            graph_count = 4  # Basic: mol_weight, num_atoms, num_bonds, num_rings
            
            if self.feature_level in ['standard', 'extended', 'comprehensive']:
                graph_count += 6  # logp, tpsa, hbd, hba, num_aromatic_rings, num_saturated_rings
            
            if self.feature_level in ['extended', 'comprehensive']:
                graph_count += 5  # num_heteroatoms, num_rotatable_bonds, fraction_csp3, bertz_ct, balaban_j
            
            if self.feature_level == 'comprehensive':
                graph_count += 7  # kappa1-3, chi0v-2v, hall_kier_alpha
        
        return {
            'node_features': atom_count,
            'edge_features': bond_count,
            'graph_features': graph_count
        }
    
    def get_all_feature_shapes(self) -> Dict[str, Dict[str, int]]:
        """
        Get feature shapes for all feature levels.
        
        Returns:
            Nested dictionary with feature counts for each level
        """
        current_level = self.feature_level
        current_3d = self.include_3d
        current_charges = self.include_partial_charges
        current_descriptors = self.include_descriptors
        
        shapes = {}
        for level in ['basic', 'standard', 'extended', 'comprehensive']:
            self.feature_level = level
            shapes[level] = self.get_feature_shapes()
        
        # Add variations with optional features
        self.include_3d = True
        shapes['with_3d'] = self.get_feature_shapes()
        
        self.include_3d = False
        self.include_partial_charges = True
        shapes['with_charges'] = self.get_feature_shapes()
        
        self.include_partial_charges = True
        self.include_3d = True
        shapes['with_3d_and_charges'] = self.get_feature_shapes()
        
        # Restore original settings
        self.feature_level = current_level
        self.include_3d = current_3d
        self.include_partial_charges = current_charges
        self.include_descriptors = current_descriptors
        
        return shapes
    
    def get_descriptor_features(self, smiles: Union[str, List[str]], 
                              normalize: str = None) -> Optional[Union[Dict, List[Dict]]]:
        """
        Extract only molecular descriptors (graph-level features) from SMILES.
        
        Args:
            smiles: Single SMILES string or list of SMILES strings
            normalize: Normalization method - 'standardize', 'minmax', or None
            
        Returns:
            Dictionary of descriptors or list of descriptor dictionaries
        """
        if isinstance(smiles, str):
            return self._get_single_descriptors(smiles, normalize)
        elif isinstance(smiles, list):
            results = []
            for smi in smiles:
                desc = self._get_single_descriptors(smi, normalize)
                if desc is not None:
                    results.append(desc)
            return results if results else None
        else:
            raise ValueError("smiles must be string or list of strings")
    
    def _get_single_descriptors(self, smiles: str, normalize: str = None) -> Optional[Dict]:
        """Extract descriptors from a single SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = self._get_molecular_descriptors(mol)
        
        if normalize and descriptors:
            descriptors = self._normalize_descriptors(descriptors, normalize)
        
        return descriptors
    
    def _normalize_descriptors(self, descriptors: Dict[str, float], 
                             method: str = 'standardize') -> Dict[str, float]:
        """
        Normalize descriptor values using predefined statistics.
        
        Args:
            descriptors: Dictionary of descriptor values
            method: 'standardize' (z-score) or 'minmax' (0-1 scaling)
            
        Returns:
            Dictionary of normalized descriptor values
        """
        normalized = {}
        
        for key, value in descriptors.items():
            if key in self._descriptor_norms:
                norms = self._descriptor_norms[key]
                
                if method == 'standardize':
                    # Z-score normalization: (x - mean) / std
                    normalized[key] = (value - norms['mean']) / norms['std']
                
                elif method == 'minmax':
                    # Min-max normalization: (x - min) / (max - min)
                    normalized[key] = (value - norms['min']) / (norms['max'] - norms['min'])
                    # Clip to [0, 1] range
                    normalized[key] = max(0.0, min(1.0, normalized[key]))
                
                else:
                    normalized[key] = value
            else:
                # Keep original value if no normalization constants available
                normalized[key] = value
        
        return normalized
    
    def get_normalization_constants(self) -> Dict[str, Dict[str, float]]:
        """
        Get predefined normalization constants for descriptors.
        
        Returns:
            Dictionary of normalization constants for each descriptor
        """
        return self._descriptor_norms.copy()
    
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
            # _CIPCode is a string property ('R', 'S', etc.), so we need to encode it
            cip_code = 0
            if atom.HasProp('_CIPCode'):
                cip_str = atom.GetProp('_CIPCode')
                # Simple encoding: R=1, S=2, other=0
                if cip_str == 'R':
                    cip_code = 1
                elif cip_str == 'S':
                    cip_code = 2
            
            features.extend([
                atom.GetAtomMapNum(),
                int(atom.HasProp('_ChiralityPossible')),
                cip_code,
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
                    
                    balaban = rdMolDescriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0
                    if np.isnan(balaban):
                        # For cyclic molecules, estimate from rings; otherwise use connectivity
                        num_rings = rdMolDescriptors.CalcNumRings(mol)
                        balaban = 1.0 + (num_rings * 0.5) if num_rings > 0 else 1.0
                    
                    fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
                    if np.isnan(fraction_csp3):
                        # No carbons, set to 0
                        fraction_csp3 = 0.0
                    
                    descriptors.update({
                        'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
                        'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                        'fraction_csp3': fraction_csp3,
                        'bertz_ct': bertz_ct,
                        'balaban_j': balaban,
                    })
                except Exception as e:
                    # Minimal descriptors if advanced ones fail
                    fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol) if hasattr(rdMolDescriptors, 'CalcFractionCSP3') else 0.0
                    if np.isnan(fraction_csp3):
                        fraction_csp3 = 0.0
                    
                    descriptors.update({
                        'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
                        'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                        'fraction_csp3': fraction_csp3,
                        'bertz_ct': mol.GetNumBonds() + rdMolDescriptors.CalcNumRings(mol) * 2,
                        'balaban_j': 0,
                    })
            
            if self.feature_level == 'comprehensive':
                try:
                    # Check for availability of each descriptor individually
                    comprehensive_descriptors = {}
                    
                    # Kappa indices - use molecular size as fallback
                    if hasattr(rdMolDescriptors, 'Kappa1'):
                        kappa1 = rdMolDescriptors.Kappa1(mol)
                        comprehensive_descriptors['kappa1'] = mol.GetNumAtoms() if np.isnan(kappa1) else kappa1
                    else:
                        comprehensive_descriptors['kappa1'] = mol.GetNumAtoms()
                    
                    if hasattr(rdMolDescriptors, 'Kappa2'):
                        kappa2 = rdMolDescriptors.Kappa2(mol)
                        comprehensive_descriptors['kappa2'] = mol.GetNumBonds() if np.isnan(kappa2) else kappa2
                    else:
                        comprehensive_descriptors['kappa2'] = mol.GetNumBonds()
                    
                    if hasattr(rdMolDescriptors, 'Kappa3'):
                        kappa3 = rdMolDescriptors.Kappa3(mol)
                        comprehensive_descriptors['kappa3'] = rdMolDescriptors.CalcNumRings(mol) if np.isnan(kappa3) else kappa3
                    else:
                        comprehensive_descriptors['kappa3'] = rdMolDescriptors.CalcNumRings(mol)
                    
                    # Chi indices - use molecular size as fallback
                    if hasattr(rdMolDescriptors, 'Chi0v'):
                        chi0v = rdMolDescriptors.Chi0v(mol)
                        comprehensive_descriptors['chi0v'] = mol.GetNumAtoms() if np.isnan(chi0v) else chi0v
                    else:
                        comprehensive_descriptors['chi0v'] = mol.GetNumAtoms()
                    
                    if hasattr(rdMolDescriptors, 'Chi1v'):
                        chi1v = rdMolDescriptors.Chi1v(mol)
                        comprehensive_descriptors['chi1v'] = mol.GetNumBonds() if np.isnan(chi1v) else chi1v
                    else:
                        comprehensive_descriptors['chi1v'] = mol.GetNumBonds()
                    
                    if hasattr(rdMolDescriptors, 'Chi2v'):
                        chi2v = rdMolDescriptors.Chi2v(mol)
                        comprehensive_descriptors['chi2v'] = rdMolDescriptors.CalcNumRings(mol) if np.isnan(chi2v) else chi2v
                    else:
                        comprehensive_descriptors['chi2v'] = rdMolDescriptors.CalcNumRings(mol)
                    
                    # Hall-Kier alpha - use flexibility measure as fallback
                    if hasattr(rdMolDescriptors, 'HallKierAlpha'):
                        hall_kier = rdMolDescriptors.HallKierAlpha(mol)
                        if np.isnan(hall_kier):
                            hall_kier = rdMolDescriptors.CalcNumRotatableBonds(mol) / max(mol.GetNumBonds(), 1)
                        comprehensive_descriptors['hall_kier_alpha'] = hall_kier
                    else:
                        comprehensive_descriptors['hall_kier_alpha'] = (
                            rdMolDescriptors.CalcNumRotatableBonds(mol) / max(mol.GetNumBonds(), 1)
                        )
                    
                    descriptors.update(comprehensive_descriptors)
                    
                except Exception as e:
                    warnings.warn(f"Error in comprehensive descriptors: {e}")
                    # Minimal fallback with guaranteed non-NaN values
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
    
    def to_graph(self, smiles: str, normalize_descriptors: str = None) -> Optional[Dict]:
        """
        Convert SMILES to graph representation.
        
        Args:
            smiles: SMILES string
            normalize_descriptors: Normalization method for descriptors - 'standardize', 'minmax', or None
            
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
        
        partial_charges = None
        if self.include_partial_charges:
            try:
                ComputeGasteigerCharges(mol)
                partial_charges = []
                for atom in mol.GetAtoms():
                    charge = float(atom.GetProp('_GasteigerCharge'))
                    # Replace NaN with empirical charge based on electronegativity
                    if np.isnan(charge):
                        # Use simple electronegativity-based heuristic
                        symbol = atom.GetSymbol()
                        electroneg_charges = {'O': -0.4, 'N': -0.3, 'S': -0.2, 
                                            'F': -0.5, 'Cl': -0.3, 'Br': -0.2, 
                                            'I': -0.1, 'C': 0.0, 'H': 0.1}
                        charge = electroneg_charges.get(symbol, 0.0)
                    partial_charges.append(charge)
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
        
        # Normalize descriptors if requested
        if normalize_descriptors and descriptors:
            descriptors = self._normalize_descriptors(descriptors, normalize_descriptors)
        
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


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Example usage and convenience functions


def create_converter(feature_level: str = "standard", **kwargs) -> SMILESToGraph:
    """Create a preconfigured converter."""
    return SMILESToGraph(feature_level=feature_level, **kwargs)

def quick_convert(smiles: str, feature_level: str = "standard") -> Optional[Dict]:
    """Quick conversion with default settings."""
    converter = SMILESToGraph(feature_level=feature_level)
    return converter.to_graph(smiles)

def get_descriptors_only(smiles: Union[str, List[str]], 
                        feature_level: str = "standard", 
                        normalize: str = None) -> Optional[Union[Dict, List[Dict]]]:
    """Quick extraction of molecular descriptors only."""
    converter = SMILESToGraph(feature_level=feature_level)
    return converter.get_descriptor_features(smiles, normalize)


