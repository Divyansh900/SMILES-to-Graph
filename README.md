# SMILES to Graph Converter - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation & Requirements](#installation--requirements)
3. [API Reference](#api-reference)
4. [Chemistry Background & Feature Relevance](#chemistry-background--feature-relevance)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Overview

The `SMILESToGraph` class is a comprehensive molecular feature extraction toolkit that converts SMILES (Simplified Molecular Input Line Entry System) strings into graph representations suitable for machine learning applications. It provides configurable feature levels, built-in normalization, and standalone descriptor extraction capabilities.

### Key Features
- **Multi-level Feature Extraction**: Four feature complexity levels (basic, standard, extended, comprehensive)
- **Built-in Normalization**: Predefined scaling factors for molecular descriptors
- **Flexible Output**: Full graph representation or descriptors-only extraction
- **Robust Error Handling**: Graceful fallbacks for missing RDKit functionalities
- **Comprehensive Documentation**: Feature names, shapes, and chemical relevance

---

## Installation & Requirements

### Required Dependencies
```bash
pip install rdkit-pypi numpy
```

### Optional Dependencies (for extended functionality)
```bash
pip install pandas matplotlib seaborn  # For data analysis and visualization
```

### Python Version
- Python 3.7+ required
- Tested with Python 3.8-3.11

---

## API Reference

### Class: `SMILESToGraph`

#### Constructor

```python
SMILESToGraph(
    feature_level: str = "standard",
    include_3d: bool = False,
    include_partial_charges: bool = False,
    include_descriptors: bool = True,
    max_atomic_num: int = 100,
    common_atoms: List[str] = None
)
```

**Parameters:**
- `feature_level` (str): Feature complexity level
  - `"basic"`: Essential structural features (6 atom features, 5 bond features)
  - `"standard"`: Basic + common chemical properties (41 atom features, 7 bond features)
  - `"extended"`: Standard + advanced descriptors (51 atom features, 9 bond features)
  - `"comprehensive"`: All available features (54 atom features, 11 bond features)

- `include_3d` (bool): Include 3D coordinates if available (adds 3 features per atom)

- `include_partial_charges` (bool): Compute Gasteiger partial charges (adds 1 feature per atom)

- `include_descriptors` (bool): Include molecular descriptors in output

- `max_atomic_num` (int): Maximum atomic number for one-hot encoding (default: 100)

- `common_atoms` (List[str]): Atoms to one-hot encode (default: ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'])

#### Core Methods

##### `to_graph(smiles: str, normalize_descriptors: str = None) -> Optional[Dict]`

Convert SMILES string to complete graph representation.

**Parameters:**
- `smiles` (str): Valid SMILES string
- `normalize_descriptors` (str, optional): Normalization method
  - `"standardize"`: Z-score normalization (mean=0, std=1)
  - `"minmax"`: Min-max scaling (range 0-1)
  - `None`: No normalization

**Returns:**
```python
{
    'node_features': np.ndarray,     # Shape: (num_atoms, num_atom_features)
    'edges': np.ndarray,             # Shape: (num_edges, 2) - edge indices
    'edge_features': np.ndarray,     # Shape: (num_edges, num_edge_features)
    'descriptors': dict,             # Molecular descriptors
    'metadata': dict                 # Feature information and configuration
}
```

**Example:**
```python
converter = SMILESToGraph(feature_level='standard')
graph = converter.to_graph("CCO", normalize_descriptors='standardize')
print(f"Atoms: {graph['node_features'].shape[0]}")
print(f"Atom features: {graph['node_features'].shape[1]}")
```

##### `get_descriptor_features(smiles: Union[str, List[str]], normalize: str = None) -> Optional[Union[Dict, List[Dict]]]`

Extract only molecular descriptors (graph-level features).

**Parameters:**
- `smiles` (str or List[str]): Single SMILES or list of SMILES strings
- `normalize` (str, optional): Normalization method ("standardize", "minmax", or None)

**Returns:**
- Single dict (if input is string) or list of dicts (if input is list)
- Each dict contains descriptor_name: value pairs

**Example:**
```python
# Single molecule
descriptors = converter.get_descriptor_features("CCO", normalize='standardize')

# Multiple molecules
smiles_list = ["CCO", "CCN", "CCC"]
desc_list = converter.get_descriptor_features(smiles_list, normalize='minmax')
```

#### Feature Information Methods

##### `get_feature_shapes() -> Dict[str, int]`

Get feature dimensions for current configuration.

**Returns:**
```python
{
    'node_features': int,    # Number of atom features
    'edge_features': int,    # Number of bond features  
    'graph_features': int    # Number of molecular descriptors
}
```

##### `get_all_feature_shapes() -> Dict[str, Dict[str, int]]`

Get feature dimensions for all configurations.

**Returns:**
```python
{
    'basic': {'node_features': 6, 'edge_features': 5, 'graph_features': 4},
    'standard': {'node_features': 41, 'edge_features': 7, 'graph_features': 10},
    'extended': {'node_features': 51, 'edge_features': 9, 'graph_features': 15},
    'comprehensive': {'node_features': 54, 'edge_features': 11, 'graph_features': 22},
    'with_3d': {...},
    'with_charges': {...},
    'with_3d_and_charges': {...}
}
```

##### `get_feature_names() -> Tuple[List[str], List[str]]`

Get human-readable feature names.

**Returns:**
- Tuple of (atom_feature_names, bond_feature_names)

##### `get_normalization_constants() -> Dict[str, Dict[str, float]]`

Get predefined normalization statistics.

**Returns:**
```python
{
    'descriptor_name': {
        'mean': float,    # For standardization
        'std': float,     # For standardization  
        'min': float,     # For min-max scaling
        'max': float      # For min-max scaling
    }
}
```

### Convenience Functions

#### `create_converter(feature_level: str = "standard", **kwargs) -> SMILESToGraph`
Factory function to create converter with specified configuration.

#### `quick_convert(smiles: str, feature_level: str = "standard", normalize_descriptors: str = None) -> Optional[Dict]`
One-line conversion with default settings.

#### `get_descriptors_only(smiles: Union[str, List[str]], feature_level: str = "standard", normalize: str = None) -> Optional[Union[Dict, List[Dict]]]`
Quick descriptor extraction without creating converter instance.

### Preset Configurations

```python
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
```

---

## Chemistry Background & Feature Relevance

### Understanding Molecular Representations

Molecules can be represented as graphs where:
- **Nodes (Atoms)**: Represent individual atoms with their properties
- **Edges (Bonds)**: Represent chemical bonds between atoms with bond properties
- **Graph-level features**: Represent overall molecular properties

### Feature Categories and Chemical Relevance

#### Atom Features

##### Basic Level Features
1. **Atomic Number**: Element identity
   - *Relevance*: Fundamental property determining chemical behavior
   - *Use Cases*: All applications, especially for distinguishing heteroatoms

2. **Formal Charge**: Charge assigned to atom in Lewis structure
   - *Relevance*: Indicates electron distribution, affects reactivity
   - *Use Cases*: Reaction prediction, electrostatic interactions

3. **Hybridization**: Orbital hybridization state (sp³, sp², sp, aromatic)
   - *Relevance*: Determines geometry and bonding patterns
   - *Use Cases*: 3D structure prediction, reaction mechanisms

4. **Aromaticity**: Whether atom is part of aromatic system
   - *Relevance*: Special stability, different reactivity patterns
   - *Use Cases*: Drug design (π-π interactions), metabolic stability

5. **Degree**: Number of bonded atoms
   - *Relevance*: Local connectivity, steric effects
   - *Use Cases*: Reactivity prediction, binding site analysis

6. **Hydrogen Count**: Total number of hydrogen atoms attached
   - *Relevance*: Affects polarity, hydrogen bonding potential
   - *Use Cases*: Solubility prediction, drug-target interactions

##### Standard Level Features (Additional)
7. **Valence Information**: Implicit and explicit valence electrons
   - *Relevance*: Electron availability for bonding
   - *Use Cases*: Reaction mechanism prediction, radical chemistry

8. **Ring Membership**: Whether atom is in a ring structure
   - *Relevance*: Conformational constraints, strain effects
   - *Use Cases*: Pharmacokinetics, synthetic accessibility

9. **Atomic Mass**: Isotope-specific mass
   - *Relevance*: Important for isotope effects, NMR prediction
   - *Use Cases*: Metabolic studies, analytical chemistry

10. **Chirality**: Stereochemical information
    - *Relevance*: Different biological activities for enantiomers
    - *Use Cases*: Drug design, chiral separation

11. **Element One-hot Encoding**: Binary flags for common atoms
    - *Relevance*: Efficient representation of element diversity
    - *Use Cases*: Deep learning models, element-specific predictions

##### Extended Level Features (Additional)
12. **Detailed Ring Analysis**: Ring sizes (3-7 membered, larger)
    - *Relevance*: Ring strain, conformational preferences
    - *Use Cases*: Synthetic chemistry, drug metabolism

13. **Ring Count**: Number of rings containing this atom
    - *Relevance*: Indicates fused ring systems, rigidity
    - *Use Cases*: Permeability prediction, synthetic complexity

14. **Radical Electrons**: Unpaired electrons
    - *Relevance*: Reactivity, instability indicators
    - *Use Cases*: Reaction mechanism studies, antioxidant design

##### Comprehensive Level Features (Additional)
15. **Atom Mapping**: Reaction mapping information
    - *Relevance*: Tracks atoms through chemical transformations
    - *Use Cases*: Reaction prediction, mechanism elucidation

16. **CIP Stereochemistry**: R/S configuration
    - *Relevance*: Precise stereochemical description
    - *Use Cases*: Chiral drug development, enantioselective synthesis

#### Bond Features

##### Basic Level Features
1. **Bond Type**: Single, double, triple, aromatic
   - *Relevance*: Determines bond strength, length, reactivity
   - *Use Cases*: All applications, fundamental chemical property

2. **Conjugation**: Whether bond is part of conjugated system
   - *Relevance*: Electron delocalization, stability
   - *Use Cases*: UV-Vis prediction, photochemistry

##### Standard Level Features (Additional)
3. **Ring Bonds**: Whether bond is part of ring structure
   - *Relevance*: Conformational constraints, ring strain
   - *Use Cases*: Conformational analysis, synthetic planning

4. **Stereochemistry**: E/Z configuration for double bonds
   - *Relevance*: Geometric isomerism affects properties
   - *Use Cases*: Isomer enumeration, property prediction

##### Extended/Comprehensive Level Features (Additional)
5. **Bond Direction**: Stereochemical wedge/dash notation
   - *Relevance*: 3D spatial arrangement information
   - *Use Cases*: 3D structure generation, chiral recognition

#### Molecular Descriptors (Graph-level Features)

##### Basic Descriptors
1. **Molecular Weight**: Sum of atomic masses
   - *Relevance*: Drug-like properties, ADMET prediction
   - *Use Cases*: Drug discovery (Rule of 5), formulation

2. **Atom/Bond Counts**: Basic structural complexity
   - *Relevance*: Size and complexity measures
   - *Use Cases*: Synthetic accessibility, screening filters

3. **Ring Count**: Number of ring systems
   - *Relevance*: Rigidity, complexity indicator
   - *Use Cases*: CNS penetration, metabolic stability

##### Standard Descriptors (Additional)
4. **LogP**: Lipophilicity measure
   - *Relevance*: Membrane permeability, distribution
   - *Use Cases*: ADMET prediction, drug design

5. **TPSA** (Topological Polar Surface Area): Polar surface area
   - *Relevance*: Blood-brain barrier penetration, absorption
   - *Use Cases*: CNS drugs, oral bioavailability

6. **Hydrogen Bonding**: Donor/acceptor counts
   - *Relevance*: Solubility, protein binding
   - *Use Cases*: Drug-target interactions, crystallization

7. **Aromatic Ring Counts**: Aromatic vs saturated rings
   - *Relevance*: Metabolic stability, π-π interactions
   - *Use Cases*: Drug metabolism prediction, binding affinity

##### Extended Descriptors (Additional)
8. **Heteroatom Count**: Non-carbon heavy atoms
   - *Relevance*: Polarity, reactivity sites
   - *Use Cases*: Metabolic prediction, toxicity assessment

9. **Rotatable Bonds**: Conformational flexibility
   - *Relevance*: Binding entropy, oral absorption
   - *Use Cases*: Drug design, molecular dynamics

10. **sp³ Carbon Fraction**: 3D character measure
    - *Relevance*: "Drug-likeness", synthetic accessibility
    - *Use Cases*: Lead optimization, natural product analysis

11. **Bertz Complexity**: Topological complexity index
    - *Relevance*: Synthetic difficulty, information content
    - *Use Cases*: Synthetic route planning, diversity analysis

12. **Balaban J Index**: Topological descriptor
    - *Relevance*: Molecular branching and connectivity
    - *Use Cases*: QSAR modeling, property prediction

##### Comprehensive Descriptors (Additional)
13. **Kappa Indices**: Shape descriptors (κ₁, κ₂, κ₃)
    - *Relevance*: Molecular shape and flexibility
    - *Use Cases*: Receptor binding, selectivity prediction

14. **Chi Indices**: Connectivity descriptors (χ₀ᵛ, χ₁ᵛ, χ₂ᵛ)
    - *Relevance*: Electron accessibility, branching
    - *Use Cases*: QSAR modeling, activity prediction

15. **Hall-Kier Alpha**: Flexibility descriptor
    - *Relevance*: Molecular flexibility and rigidity
    - *Use Cases*: Binding kinetics, conformational analysis

### Application-Specific Feature Selection

#### Drug Discovery Applications
- **Primary Features**: LogP, TPSA, HBD/HBA, molecular weight, aromatic rings
- **Secondary Features**: Rotatable bonds, sp³ fraction, formal charges
- **Rationale**: Focus on ADMET properties and drug-likeness filters

#### Reaction Prediction
- **Primary Features**: Partial charges, bond types, formal charges, atom mapping
- **Secondary Features**: Hybridization, radical electrons, stereochemistry
- **Rationale**: Emphasize electron distribution and reactivity indicators

#### Property Prediction (Physical/Chemical)
- **Primary Features**: Topological descriptors, connectivity indices, complexity measures
- **Secondary Features**: Ring information, branching descriptors
- **Rationale**: Capture structural features that correlate with bulk properties

#### Toxicity Assessment
- **Primary Features**: Heteroatom counts, aromatic rings, reactive groups
- **Secondary Features**: Electrophilic centers, metabolic soft spots
- **Rationale**: Identify structural alerts and reactive functionalities

### Feature Normalization Guidelines

#### When to Use Standardization (Z-score)
- **Machine Learning Models**: SVMs, neural networks, linear regression
- **Mixed Feature Types**: When combining categorical and continuous features
- **Outlier Sensitivity**: When outliers should be preserved but scaled

#### When to Use Min-Max Scaling
- **Neural Networks**: Especially with sigmoid/tanh activations
- **Distance-Based Methods**: k-NN, clustering algorithms
- **Bounded Outputs**: When features should be in [0,1] range

#### Feature-Specific Considerations
- **Count Features**: Usually benefit from log-transformation before scaling
- **Ratio Features**: Often naturally bounded, may not need normalization  
- **Binary Features**: Typically left unscaled
- **Highly Skewed Features**: Consider log or power transformations

### Chemical Intuition for Feature Engineering

#### Pharmacokinetic Relevance
1. **Absorption**: TPSA, HBD/HBA, molecular weight, rotatable bonds
2. **Distribution**: LogP, protein binding (aromatic rings, charges)
3. **Metabolism**: sp³ fraction, reactive groups, aromatic rings
4. **Excretion**: Molecular weight, polarity, charge state

#### Protein-Ligand Interactions
1. **Shape Complementarity**: Kappa indices, molecular volume
2. **Electrostatic Interactions**: Formal charges, partial charges
3. **Hydrophobic Interactions**: LogP, aromatic carbon count
4. **Hydrogen Bonding**: HBD/HBA, polar surface area

#### Chemical Reactivity
1. **Electrophilic Sites**: Formal charges, electron-withdrawing groups
2. **Nucleophilic Sites**: Lone pairs, electron density
3. **Radical Reactivity**: Unpaired electrons, bond dissociation energies
4. **Pericyclic Reactions**: Conjugation patterns, frontier orbitals

---

## Usage Examples

### Basic Usage

```python
from smiles_to_graph import SMILESToGraph, quick_convert

# Quick conversion with defaults
graph = quick_convert("CCO")  # Ethanol
print(f"Node features shape: {graph['node_features'].shape}")
print(f"Edge features shape: {graph['edge_features'].shape}")
```

### Feature Level Comparison

```python
converter = SMILESToGraph()
smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"  # Ibuprofen

# Compare feature levels
for level in ['basic', 'standard', 'extended', 'comprehensive']:
    converter.feature_level = level
    shapes = converter.get_feature_shapes()
    print(f"{level.title()}: {shapes}")
```

### Descriptor Extraction and Analysis

```python
# Extract descriptors for multiple molecules
drug_smiles = [
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",            # Aspirin  
    "CCN(CC)CC(=O)Nc1c(C)cccc1C"        # Lidocaine
]

converter = SMILESToGraph(feature_level='extended')

# Get raw and normalized descriptors
raw_descriptors = converter.get_descriptor_features(drug_smiles)
norm_descriptors = converter.get_descriptor_features(drug_smiles, normalize='standardize')

# Compare molecular properties
for i, name in enumerate(['Ibuprofen', 'Aspirin', 'Lidocaine']):
    print(f"\n{name}:")
    print(f"  MW: {raw_descriptors[i]['mol_weight']:.1f}")
    print(f"  LogP: {raw_descriptors[i]['logp']:.2f}")
    print(f"  TPSA: {raw_descriptors[i]['tpsa']:.1f}")
```

### Custom Configuration for Drug Discovery

```python
# Drug discovery focused configuration
drug_converter = SMILESToGraph(
    feature_level='extended',
    include_partial_charges=True,
    include_descriptors=True,
    common_atoms=['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br']
)

# Process with normalization for ML pipeline
smiles = "Cc1ccc(cc1)C(=O)c1ccccc1"  # Benzophenone derivative
graph = drug_converter.to_graph(smiles, normalize_descriptors='standardize')

# Extract features for model training
X_nodes = graph['node_features']
X_edges = graph['edge_features']  
X_graph = list(graph['descriptors'].values())

print(f"Ready for ML: Nodes {X_nodes.shape}, Edges {X_edges.shape}, Graph {len(X_graph)} features")
```

### Batch Processing Pipeline

```python
import pandas as pd
import numpy as np

def process_smiles_dataset(smiles_list, output_format='descriptors'):
    """
    Process a dataset of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        output_format: 'descriptors', 'graphs', or 'both'
    """
    converter = SMILESToGraph(feature_level='standard')
    results = []
    failed = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            if output_format in ['descriptors', 'both']:
                desc = converter.get_descriptor_features(smiles, normalize='standardize')
                if desc:
                    desc['smiles'] = smiles
                    desc['index'] = i
                    results.append(desc)
                else:
                    failed.append((i, smiles, 'Invalid SMILES'))
            
            if output_format in ['graphs', 'both']:
                graph = converter.to_graph(smiles, normalize_descriptors='standardize')
                if graph:
                    # Store graph data separately or combine with descriptors
                    pass
                
        except Exception as e:
            failed.append((i, smiles, str(e)))
    
    print(f"Processed: {len(results)}/{len(smiles_list)} molecules")
    if failed:
        print(f"Failed: {len(failed)} molecules")
        for idx, smi, error in failed[:5]:  # Show first 5 failures
            print(f"  {idx}: {smi} - {error}")
    
    return pd.DataFrame(results), failed

# Example usage
smiles_dataset = ["CCO", "CCN", "CCC", "INVALID", "c1ccccc1"]
df, failures = process_smiles_dataset(smiles_dataset)
print(df.head())
```

### Feature Analysis and Selection

```python
def analyze_feature_importance(smiles_list, target_property):
    """
    Analyze which molecular features correlate with target property.
    """
    converter = SMILESToGraph(feature_level='extended')
    
    # Get all descriptors
    descriptors_list = converter.get_descriptor_features(smiles_list, normalize='standardize')
    
    # Convert to DataFrame
    df = pd.DataFrame(descriptors_list)
    df['target'] = target_property
    
    # Calculate correlations
    correlations = df.corr()['target'].drop('target').abs().sort_values(ascending=False)
    
    print("Top 10 features correlated with target:")
    for feature, corr in correlations.head(10).items():
        print(f"  {feature}: {corr:.3f}")
    
    return correlations

# Example: Analyze features for solubility prediction
# (You would provide actual solubility values)
smiles = ["CCO", "CCCCCCCC", "c1ccccc1O", "CC(=O)O"]
solubility = [0.8, 0.1, 0.6, 1.0]  # Example values
correlations = analyze_feature_importance(smiles, solubility)
```

---

## Best Practices

### 1. Feature Level Selection
- **Start with 'standard'**: Good balance of information and computational efficiency
- **Use 'basic' for**: Large datasets, fast prototyping, limited computational resources
- **Use 'extended' for**: Drug discovery, property prediction, detailed analysis
- **Use 'comprehensive' for**: Research applications, maximum information extraction

### 2. Normalization Strategy
- **Always normalize descriptors** for machine learning applications
- **Use standardization** for most ML algorithms (SVM, neural networks, linear models)
- **Use min-max scaling** for algorithms sensitive to feature ranges (some neural networks)
- **Test both methods** with cross-validation to determine optimal approach

### 3. Error Handling
```python
def robust_conversion(smiles_list):
    converter = SMILESToGraph()
    results = []
    
    for smiles in smiles_list:
        try:
            graph = converter.to_graph(smiles)
            if graph is not None:
                results.append(graph)
            else:
                print(f"Failed to parse: {smiles}")
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
    
    return results
```

### 4. Memory Management for Large Datasets
```python
def batch_process_large_dataset(smiles_list, batch_size=1000):
    converter = SMILESToGraph(feature_level='standard')
    
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        
        # Process batch
        descriptors = converter.get_descriptor_features(batch, normalize='standardize')
        
        # Save to disk or process immediately
        # This prevents memory buildup
        yield descriptors
```

### 5. Feature Engineering Tips
- **Combine multiple feature levels** for ensemble methods
- **Engineer domain-specific features** based on chemical knowledge
- **Use feature selection** to reduce dimensionality
- **Validate chemical relevance** of important features

### 6. Validation and Quality Control
```python
def validate_features(graph_data):
    """Validate extracted features for quality control."""
    checks = {
        'non_negative_counts': True,
        'reasonable_ranges': True,
        'missing_values': False
    }
    
    # Check for negative counts where inappropriate
    descriptors = graph_data['descriptors']
    count_features = ['num_atoms', 'num_bonds', 'num_rings']
    
    for feat in count_features:
        if feat in descriptors and descriptors[feat] < 0:
            checks['non_negative_counts'] = False
    
    # Check for missing values
    if any(np.isnan(val) for val in descriptors.values()):
        checks['missing_values'] = True
    
    return checks
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. RDKit Import Errors
```
Error: "Cannot import rdkit"
Solution: pip install rdkit-pypi
Alternative: conda install -c conda-forge rdkit
```

#### 2. Invalid SMILES Handling
```python
# Problem: SMILES parsing fails
smiles = "INVALID_SMILES"
graph = converter.to_graph(smiles)  # Returns None

# Solution: Always check return values
if graph is not None:
    # Process graph
    pass
else:
    print(f"Invalid SMILES: {smiles}")
```

#### 3. Missing Descriptor Functions
```
Warning: "Error in comprehensive descriptors: 'module' object has no attribute 'BertzCT'"
```
This occurs with older RDKit versions. The code automatically falls back to alternative calculations.

#### 4. Memory Issues with Large Datasets
```python
# Problem: Out of memory with large datasets
# Solution: Use batch processing
def memory_efficient_processing(smiles_list):
    converter = SMILESToGraph()
    
    # Process one at a time, don't store all in memory
    for smiles in smiles_list:
        graph = converter.to_graph(smiles)
        if graph:
            # Process immediately, don't accumulate
            yield process_single_graph(graph)
```

#### 5. Feature Dimension Mismatches
```python
# Problem: Inconsistent feature dimensions across molecules
# Solution: Always check feature shapes
shapes = converter.get_feature_shapes()
expected_node_features = shapes['node_features']

graph = converter.to_graph(smiles)
actual_features = graph['node_features'].shape[1]

assert actual_features == expected_node_features, f"Feature mismatch: expected {expected_node_features}, got {actual_features}"
```

#### 6. Normalization Issues
```python
# Problem: Descriptors have unexpected values after normalization
descriptors = converter.get_descriptor_features(smiles, normalize='standardize')

# Solution: Check normalization constants
norms = converter.get_normalization_constants()
for desc_name, value in descriptors.items():
    if desc_name in norms:
        # Verify normalization is reasonable
        expected_range = [-3, 3]  # Typical standardized range
        if not (expected_range[0] <= value <= expected_range[1]):
            print(f"Warning: {desc_name} = {value} outside expected range")
```

### Performance Optimization

#### 1. Feature Level Selection
```python
# For large-scale screening (>10K molecules)
converter = SMILESToGraph(feature_level='basic')

# For detailed analysis (<1K molecules)  
converter = SMILESToGraph(feature_level='comprehensive')
```

#### 2. Disable Unnecessary Features
```python
# If you don't need 3D or partial charges
converter = SMILESToGraph(
    feature_level='standard',
    include_3d=False,  # Saves computation
    include_partial_charges=False,  # Saves computation
    include_descriptors=True  # Keep if needed
)
```

#### 3. Precompute Feature Shapes
```python
# Compute once, reuse many times
shapes = converter.get_feature_shapes()
node_dim = shapes['node_features']

# Use in model initialization
model = GraphNeuralNetwork(node_dim=node_dim, ...)
```

### Debugging Tools

#### 1. Feature Name Inspection
```python
atom_names, bond_names = converter.get_feature_names()
print(f"Atom features ({len(atom_names)}):")
for i, name in enumerate(atom_names):
    print(f"  {i}: {name}")
```

#### 2. Descriptor Value Ranges
```python
def check_descriptor_ranges(smiles_list):
    converter = SMILESToGraph(feature_level='extended')
    all_descriptors = converter.get_descriptor_features(smiles_list)
    
    df = pd.DataFrame(all_descriptors)
    print("Descriptor ranges:")
    print(
