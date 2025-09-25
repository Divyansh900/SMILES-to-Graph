# SMILES to Graph Converter - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation and Dependencies](#installation-and-dependencies)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Feature Levels](#feature-levels)
6. [Chemistry Background](#chemistry-background)
7. [Feature Reference](#feature-reference)
8. [Use Case Guidelines](#use-case-guidelines)
9. [Examples](#examples)
10. [Performance Considerations](#performance-considerations)
11. [Troubleshooting](#troubleshooting)

## Overview

The SMILES to Graph Converter transforms molecular SMILES (Simplified Molecular Input Line Entry System) strings into graph-structured data suitable for machine learning applications. The converter provides multiple feature levels, from basic structural information to comprehensive chemical descriptors, enabling optimization for different computational chemistry tasks.

### Key Features
- **Configurable Feature Complexity**: Four levels from basic to comprehensive
- **Fixed-Size Representations**: Consistent feature dimensions within each level
- **Domain-Specific Presets**: Optimized configurations for common applications
- **Chemical Interpretability**: Features grounded in chemical theory
- **Performance Optimization**: Scalable from minimal to exhaustive feature extraction

## Installation and Dependencies

```bash
pip install rdkit-pypi numpy
```

Required packages:
- `rdkit`: Molecular informatics toolkit
- `numpy`: Numerical computing
- `warnings`: Error handling (built-in)
- `typing`: Type annotations (built-in)

## Quick Start

```python
from smiles_to_graph import SMILESToGraph, quick_convert

# Simple conversion
graph = quick_convert("CCO", feature_level="standard")
print(f"Atoms: {graph['node_features'].shape}")
print(f"Bonds: {graph['edge_features'].shape}")

# Custom converter
converter = SMILESToGraph(
    feature_level="extended",
    include_partial_charges=True
)
graph = converter.to_graph("c1ccccc1")  # benzene
```



### Compatibility Features

The converter includes robust handling for different RDKit versions:

#### Automatic Fallbacks
The system automatically detects missing descriptors and provides chemically meaningful alternatives:

**Missing Descriptors Handled**:
- `BertzCT` → `num_bonds + num_rings * 2` (complexity proxy)
- `Kappa1/2/3` → Structural counts (atoms, bonds, rings)
- `Chi0v/1v/2v` → Connectivity measures
- `HallKierAlpha` → `rotatable_bonds / total_bonds` (flexibility measure)

**Valence API Compatibility**:
- Uses `GetTotalValence()` for cross-version compatibility
- Approximates implicit valence as `total_valence - formal_charge`
- Handles deprecated valence methods gracefully

#### Version Detection
```python
# The converter automatically detects available features:
converter = SMILESToGraph(feature_level="comprehensive")
graph = converter.to_graph("CCO")

# Check which descriptors were used vs. fallbacks in metadata
print(graph['metadata'])  # Shows actual feature configuration used
```

## API Reference

### SMILESToGraph Class

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

**Parameters**:
- `feature_level`: Feature complexity ('basic', 'standard', 'extended', 'comprehensive')
- `include_3d`: Include 3D coordinates if available
- `include_partial_charges`: Compute Gasteiger partial charges
- `include_descriptors`: Include molecular-level descriptors
- `max_atomic_num`: Maximum atomic number for one-hot encoding
- `common_atoms`: List of atoms for one-hot encoding

#### Methods

##### `to_graph(smiles: str) -> Optional[Dict]`
Convert SMILES string to graph representation.

**Returns**:
```python
{
    'node_features': np.ndarray,     # Shape: (n_atoms, n_atom_features)
    'edges': np.ndarray,             # Shape: (n_edges, 2)
    'edge_features': np.ndarray,     # Shape: (n_edges, n_edge_features)
    'descriptors': Dict[str, float], # Molecular descriptors
    'metadata': Dict                 # Conversion metadata
}
```

##### `get_feature_names() -> Tuple[List[str], List[str]]`
Get feature names for current configuration.

**Returns**: `(atom_feature_names, bond_feature_names)`

### Convenience Functions

#### `create_converter(feature_level: str = "standard", **kwargs) -> SMILESToGraph`
Create preconfigured converter.

#### `quick_convert(smiles: str, feature_level: str = "standard") -> Optional[Dict]`
One-line conversion with default settings.

### Preset Configurations

#### `PRESET_CONFIGS`
Dictionary containing optimized configurations:
- `'drug_discovery'`: ADMET-focused features
- `'reaction_prediction'`: Reaction mechanism features  
- `'property_prediction'`: General property modeling
- `'minimal'`: Lightweight processing

### Compatibility Features

The converter includes robust handling for different RDKit versions:

#### Automatic Fallbacks
The system automatically detects missing descriptors and provides chemically meaningful alternatives:

**Missing Descriptors Handled**:
- `BertzCT` → `num_bonds + num_rings * 2` (complexity proxy)
- `Kappa1/2/3` → Structural counts (atoms, bonds, rings)
- `Chi0v/1v/2v` → Connectivity measures
- `HallKierAlpha` → `rotatable_bonds / total_bonds` (flexibility measure)

**Valence API Compatibility**:
- Uses `GetTotalValence()` for cross-version compatibility
- Approximates implicit valence as `total_valence - formal_charge`
- Handles deprecated valence methods gracefully

#### Version Detection
```python
# The converter automatically detects available features:
converter = SMILESToGraph(feature_level="comprehensive")
graph = converter.to_graph("CCO")

# Check which descriptors were used vs. fallbacks in metadata
print(graph['metadata'])  # Shows actual feature configuration used
```

## Feature Levels

### Basic Level
**Purpose**: Minimal structural representation  
**Use Cases**: Fast preprocessing, baseline models, memory-constrained environments  
**Atom Features**: 6 features per atom  
**Bond Features**: 5 features per bond  

Features include essential structural information: atomic number, charge, hybridization, aromaticity, connectivity.

### Standard Level
**Purpose**: Balanced chemical representation  
**Use Cases**: General property prediction, drug discovery, most ML applications  
**Atom Features**: ~32 features per atom  
**Bond Features**: 7 features per bond  

Adds one-hot encodings, valence information, mass, chirality, and ring membership.

### Extended Level
**Purpose**: Detailed chemical characterization  
**Use Cases**: Complex property prediction, reaction outcome prediction, detailed SAR studies  
**Atom Features**: ~40 features per atom  
**Bond Features**: 9 features per bond  

Includes ring analysis, advanced connectivity measures, and chemical complexity descriptors.

### Comprehensive Level
**Purpose**: Maximum chemical information  
**Use Cases**: Research applications, reaction mechanism studies, detailed molecular analysis  
**Atom Features**: ~45 features per atom  
**Bond Features**: 11 features per bond  

Incorporates topological indices, advanced stereochemistry, and complete molecular descriptors.

## Feature Reference

### Atom Features

#### Basic Features (All Levels)
| Feature | Description | Chemistry Relevance |
|---------|-------------|-------------------|
| `atomic_num` | Atomic number (1-118) | Element identity, fundamental chemical behavior |
| `formal_charge` | Formal charge (-3 to +3 typical) | Electrostatic interactions, reactivity |
| `hybridization` | sp, sp², sp³ hybridization | Geometry, bond angles, reactivity |
| `is_aromatic` | Aromatic system membership | Stability, π-electron effects, planarity |
| `degree` | Number of bonded neighbors | Steric hindrance, coordination |
| `total_h` | Total hydrogen count | Polarity, hydrogen bonding |

#### Standard Additional Features
| Feature | Description | Chemistry Relevance |
|---------|-------------|-------------------|
| `implicit_valence` | Implicit valence electrons | Bonding capacity |
| `explicit_valence` | Explicit valence electrons | Actual bonding state |
| `is_in_ring` | Ring system membership | Conformational constraint, stability |
| `mass` | Atomic mass | Isotope effects, kinetics |
| `is_chiral` | Chirality center | Stereochemistry, biological activity |
| `total_degree` | Total connectivity | Molecular branching |
| One-hot encodings | Element-specific flags | Element-specific properties |

#### Extended Additional Features
| Feature | Description | Chemistry Relevance |
|---------|-------------|-------------------|
| Ring size features | 3-7+ membered rings | Ring strain, conformational preferences |
| `num_rings` | Number of rings containing atom | Rigidity, aromaticity extent |
| `total_valence` | Complete valence state | Electron availability |
| `num_radical_electrons` | Unpaired electrons | Reactivity, stability |

#### Comprehensive Additional Features
| Feature | Description | Chemistry Relevance |
|---------|-------------|-------------------|
| `atom_map_num` | Reaction mapping identifier | Reaction mechanism tracking |
| Topological indices | Graph-theoretic descriptors | Molecular complexity, similarity |
| Advanced chirality | Complete stereochemical state | Enantioselectivity |

#### Optional Features
| Feature | Description | When to Use |
|---------|-------------|-------------|
| `partial_charge` | Gasteiger partial charges | Electrostatic modeling, reactivity prediction |
| 3D coordinates | x, y, z positions | Conformational analysis, 3D-QSAR |

### Bond Features

#### Basic Features (All Levels)
| Feature | Description | Chemistry Relevance |
|---------|-------------|-------------------|
| `single/double/triple` | Bond order | Strength, length, reactivity |
| `aromatic` | Aromatic bond | Delocalization, stability |
| `conjugated` | Conjugation participation | Electronic delocalization |

#### Standard Additional Features
| Feature | Description | Chemistry Relevance |
|---------|-------------|-------------------|
| `is_in_ring` | Ring bond | Conformational constraint |
| `has_stereo` | Stereochemical information | E/Z isomerism |

#### Extended Additional Features
| Feature | Description | Chemistry Relevance |
|---------|-------------|-------------------|
| `valence_contrib` | Valence contribution to atoms | Electron density distribution |

#### Comprehensive Additional Features
| Feature | Description | Chemistry Relevance |
|---------|-------------|-------------------|
| `has_direction` | Directional bonding | Advanced stereochemistry |
| `bond_type_double` | Precise bond order | Fractional bond orders |

### Molecular Descriptors

#### Basic Descriptors (Standard+)
| Descriptor | Range | Chemistry Relevance |
|------------|-------|-------------------|
| `mol_weight` | 1-2000+ Da | Size, dosing, membrane permeability |
| `logp` | -3 to +8 | Lipophilicity, ADMET properties |
| `tpsa` | 0-300+ Ų | Polar surface area, blood-brain barrier |
| `hbd/hba` | 0-20+ | Hydrogen bonding, solubility |

#### Extended Descriptors
| Descriptor | Description | Applications |
|------------|-------------|-------------|
| `num_rotatable_bonds` | Conformational flexibility | Bioavailability, binding entropy |
| `fraction_csp3` | Saturation level | Drug-likeness, synthetic accessibility |
| `bertz_ct` | Structural complexity | Synthetic difficulty |

#### Comprehensive Descriptors (Version-Safe)
The comprehensive level now includes individual checks for each topological descriptor:

| Descriptor | Available Check | Fallback Value | Description |
|------------|----------------|----------------|-------------|
| `kappa1` | `hasattr(rdMolDescriptors, 'Kappa1')` | `mol.GetNumAtoms()` | Shape index 1 → atom count |
| `kappa2` | `hasattr(rdMolDescriptors, 'Kappa2')` | `mol.GetNumBonds()` | Shape index 2 → bond count |
| `kappa3` | `hasattr(rdMolDescriptors, 'Kappa3')` | `CalcNumRings(mol)` | Shape index 3 → ring count |
| `chi0v` | `hasattr(rdMolDescriptors, 'Chi0v')` | `mol.GetNumAtoms()` | Valence connectivity 0 → atom count |
| `chi1v` | `hasattr(rdMolDescriptors, 'Chi1v')` | `mol.GetNumBonds()` | Valence connectivity 1 → bond count |
| `chi2v` | `hasattr(rdMolDescriptors, 'Chi2v')` | `CalcNumRings(mol)` | Valence connectivity 2 → ring count |
| `hall_kier_alpha` | `hasattr(rdMolDescriptors, 'HallKierAlpha')` | `rotatable_bonds / total_bonds` | Flexibility index |

#### Error Handling Strategy
The converter uses a graceful degradation approach:
1. **Individual Detection**: Each descriptor checked separately
2. **Silent Fallbacks**: Missing descriptors replaced with meaningful alternatives
3. **Chemical Relevance**: Fallback values maintain interpretability
4. **Consistency**: Same feature dimensions regardless of RDKit version

## Chemistry Background

### Molecular Graph Theory
Molecules are naturally represented as graphs where:
- **Nodes (Vertices)**: Atoms with associated properties
- **Edges**: Chemical bonds with bond orders and types
- **Graph Properties**: Molecular descriptors and topological indices

### Key Chemical Concepts

#### Aromaticity
Aromatic systems exhibit special stability due to π-electron delocalization. Features:
- Planar ring structure
- 4n+2 π electrons (Hückel's rule)
- Equal bond lengths
- Special reactivity patterns

**ML Relevance**: Aromatic atoms/bonds behave differently in reactions and have distinct physicochemical properties.

#### Hybridization
Atomic orbital mixing determining molecular geometry:
- **sp³**: Tetrahedral (109.5°), saturated carbons
- **sp²**: Trigonal planar (120°), alkenes, carbonyls
- **sp**: Linear (180°), alkynes, nitriles

**ML Relevance**: Determines 3D structure, reactivity, and bonding patterns.

#### Chirality
Three-dimensional arrangement around asymmetric centers:
- **R/S**: Absolute configuration
- **E/Z**: Double bond stereochemistry
- Biological activity often depends on chirality

**ML Relevance**: Critical for drug design, biological activity prediction.

#### Ring Systems
Cyclic structures with unique properties:
- **3-4 rings**: High strain, reactive
- **5-6 rings**: Stable, common in drugs
- **7+ rings**: Flexible, medium rings challenging to synthesize

**ML Relevance**: Ring size affects stability, reactivity, and synthetic accessibility.

#### Partial Charges
Electron density distribution:
- **Gasteiger charges**: Empirical method for charge calculation
- **Electronegativity**: Tendency to attract electrons
- Affects reactivity, binding, and physicochemical properties

**ML Relevance**: Essential for electrostatic modeling and reactivity prediction.

### Molecular Descriptors

#### Lipinski's Rule of Five
Drug-like properties criteria:
- Molecular weight ≤ 500 Da
- LogP ≤ 5
- Hydrogen bond donors ≤ 5
- Hydrogen bond acceptors ≤ 10

#### ADMET Properties
- **Absorption**: Gastrointestinal uptake
- **Distribution**: Tissue distribution
- **Metabolism**: Enzymatic transformation
- **Excretion**: Elimination from body
- **Toxicity**: Adverse effects

#### Topological Indices
Graph-theory descriptors:
- **Wiener Index**: Sum of shortest paths
- **Randić Index**: Connectivity-based
- **Kappa Indices**: Shape descriptors

## Use Case Guidelines

### Drug Discovery and ADMET Prediction

**Recommended Configuration**:
```python
config = {
    'feature_level': 'extended',
    'include_partial_charges': True,
    'include_descriptors': True,
    'common_atoms': ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br']
}
```

**Key Features**:
- Molecular descriptors for Lipinski compliance
- Partial charges for binding prediction
- Ring analysis for metabolic stability
- Heteroatom encoding for pharmacophores

**Applications**:
- Bioavailability prediction
- Toxicity screening
- Target binding affinity
- Metabolic stability

### Chemical Reaction Prediction

**Recommended Configuration**:
```python
config = {
    'feature_level': 'comprehensive',
    'include_partial_charges': True,
    'include_descriptors': False,
    'common_atoms': ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
}
```

**Key Features**:
- Complete stereochemical information
- Partial charges for electrophilic/nucleophilic sites
- Comprehensive atom/bond features
- Minimal molecular descriptors (focus on local features)

**Applications**:
- Reaction outcome prediction
- Mechanism elucidation
- Selectivity prediction
- Catalyst design

### Molecular Property Prediction

**Recommended Configuration**:
```python
config = {
    'feature_level': 'standard',
    'include_descriptors': True,
    'include_partial_charges': False
}
```

**Key Features**:
- Balanced feature set
- Rich molecular descriptors
- Standard chemical representations
- Good performance/interpretability trade-off

**Applications**:
- Solubility prediction
- Melting point prediction
- Spectroscopic property prediction
- General QSAR modeling

### Large-Scale Screening

**Recommended Configuration**:
```python
config = {
    'feature_level': 'basic',
    'include_descriptors': False,
    'include_partial_charges': False
}
```

**Key Features**:
- Minimal computational cost
- Fast processing
- Essential structural information
- Memory efficient

**Applications**:
- Virtual screening
- Chemical space exploration
- Preliminary filtering
- Baseline model development

### Materials Science

**Recommended Configuration**:
```python
config = {
    'feature_level': 'extended',
    'include_3d': True,
    'include_partial_charges': True,
    'common_atoms': ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si', 'B']
}
```

**Key Features**:
- Extended atom types (metalloids)
- 3D structural information
- Electronic properties
- Advanced connectivity measures

**Applications**:
- Polymer property prediction
- Catalyst design
- Electronic material properties
- Surface chemistry

## API Reference

### SMILESToGraph Class

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

**Parameters**:
- `feature_level`: Feature complexity ('basic', 'standard', 'extended', 'comprehensive')
- `include_3d`: Include 3D coordinates if available
- `include_partial_charges`: Compute Gasteiger partial charges
- `include_descriptors`: Include molecular-level descriptors
- `max_atomic_num`: Maximum atomic number for one-hot encoding
- `common_atoms`: List of atoms for one-hot encoding

#### Methods

##### `to_graph(smiles: str) -> Optional[Dict]`
Convert SMILES string to graph representation.

**Returns**:
```python
{
    'node_features': np.ndarray,     # Shape: (n_atoms, n_atom_features)
    'edges': np.ndarray,             # Shape: (n_edges, 2)
    'edge_features': np.ndarray,     # Shape: (n_edges, n_edge_features)
    'descriptors': Dict[str, float], # Molecular descriptors
    'metadata': Dict                 # Conversion metadata
}
```

##### `get_feature_names() -> Tuple[List[str], List[str]]`
Get feature names for current configuration.

**Returns**: `(atom_feature_names, bond_feature_names)`

### Convenience Functions

#### `create_converter(feature_level: str = "standard", **kwargs) -> SMILESToGraph`
Create preconfigured converter.

#### `quick_convert(smiles: str, feature_level: str = "standard") -> Optional[Dict]`
One-line conversion with default settings.

### Preset Configurations

#### `PRESET_CONFIGS`
Dictionary containing optimized configurations:
- `'drug_discovery'`: ADMET-focused features
- `'reaction_prediction'`: Reaction mechanism features  
- `'property_prediction'`: General property modeling
- `'minimal'`: Lightweight processing

## Examples

### Basic Usage

```python
from smiles_to_graph import SMILESToGraph

# Initialize converter
converter = SMILESToGraph(feature_level="standard")

# Convert molecule
smiles = "CCO"  # ethanol
graph = converter.to_graph(smiles)

print(f"Number of atoms: {graph['node_features'].shape[0]}")
print(f"Number of bonds: {graph['edges'].shape[0]}")
print(f"Molecular weight: {graph['descriptors']['mol_weight']:.2f}")
```

### Drug Discovery Pipeline

```python
from smiles_to_graph import SMILESToGraph, PRESET_CONFIGS

# Drug discovery configuration
converter = SMILESToGraph(**PRESET_CONFIGS['drug_discovery'])

# Process drug-like molecules
drug_smiles = [
    "CCO",                    # ethanol
    "CC(=O)Oc1ccccc1C(=O)O", # aspirin
    "CN1CCC2=C(C1)C(=CC=C2)O" # pramipexole core
]

for smiles in drug_smiles:
    graph = converter.to_graph(smiles)
    desc = graph['descriptors']
    
    # Check Lipinski compliance
    mw_ok = desc['mol_weight'] <= 500
    logp_ok = desc['logp'] <= 5
    hbd_ok = desc['hbd'] <= 5
    hba_ok = desc['hba'] <= 10
    
    lipinski_compliant = all([mw_ok, logp_ok, hbd_ok, hba_ok])
    print(f"{smiles}: Lipinski compliant = {lipinski_compliant}")
```

### Batch Processing

```python
import numpy as np
from smiles_to_graph import SMILESToGraph

def process_smiles_batch(smiles_list, converter):
    """Process multiple SMILES with consistent feature dimensions."""
    graphs = []
    
    for smiles in smiles_list:
        graph = converter.to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
        else:
            print(f"Failed to process: {smiles}")
    
    return graphs

# Example batch processing
smiles_batch = ["CCO", "c1ccccc1", "CC(=O)O", "CCCCC"]
converter = SMILESToGraph(feature_level="standard")
graphs = process_smiles_batch(smiles_batch, converter)

# Feature dimensions are consistent
for i, graph in enumerate(graphs):
    print(f"Molecule {i}: {graph['node_features'].shape[1]} atom features")
```

### Custom Feature Analysis

```python
from smiles_to_graph import SMILESToGraph

# Get feature names
converter = SMILESToGraph(feature_level="extended", include_partial_charges=True)
atom_names, bond_names = converter.get_feature_names()

print("Atom features:")
for i, name in enumerate(atom_names[:10]):  # First 10 features
    print(f"  {i}: {name}")

# Analyze specific molecule
graph = converter.to_graph("c1ccc2c(c1)cc(cc2)N")  # 2-naphthylamine
node_features = graph['node_features']

# Find aromatic carbons
aromatic_idx = atom_names.index('is_aromatic')
aromatic_atoms = node_features[:, aromatic_idx] == 1
print(f"Aromatic atoms: {np.sum(aromatic_atoms)}")
```

### Comparing Feature Levels (Version-Safe)

```python
from smiles_to_graph import SMILESToGraph

smiles = "CCc1ccc(cc1)N(C)C"  # N,N-dimethyl-4-ethylaniline

# Compare feature levels with automatic fallback handling
for level in ['basic', 'standard', 'extended', 'comprehensive']:
    converter = SMILESToGraph(feature_level=level)
    graph = converter.to_graph(smiles)
    
    n_atom_feat = graph['node_features'].shape[1]
    n_bond_feat = graph['edge_features'].shape[1] if graph['edge_features'].size > 0 else 0
    
    print(f"{level:12}: {n_atom_feat:2d} atom features, {n_bond_feat:2d} bond features")
    
    # Check if any fallbacks were used (comprehensive level)
    if level == 'comprehensive' and 'descriptors' in graph:
        desc = graph['descriptors']
        print(f"  Descriptors: {len(desc)} available")
```

### Version Compatibility Testing

```python
from smiles_to_graph import SMILESToGraph
from rdkit.Chem import rdMolDescriptors

# Test which descriptors are available in your RDKit version
test_descriptors = [
    'BertzCT', 'Kappa1', 'Kappa2', 'Kappa3', 
    'Chi0v', 'Chi1v', 'Chi2v', 'HallKierAlpha'
]

print("RDKit Descriptor Availability:")
for desc in test_descriptors:
    available = hasattr(rdMolDescriptors, desc)
    status = "✓ Available" if available else "✗ Missing (fallback used)"
    print(f"  {desc:15}: {status}")

# Test converter with comprehensive features
converter = SMILESToGraph(feature_level="comprehensive")
graph = converter.to_graph("CCO")
print(f"\nSuccessfully processed with {len(graph['descriptors'])} descriptors")
```
```

## Performance Considerations

### Memory Usage
Feature levels have different memory footprints:
- **Basic**: ~50 bytes per atom
- **Standard**: ~200 bytes per atom  
- **Extended**: ~250 bytes per atom
- **Comprehensive**: ~300 bytes per atom

### Processing Speed
Approximate processing times (per molecule):
- **Basic**: 0.1-0.5 ms
- **Standard**: 0.5-2 ms
- **Extended**: 2-5 ms  
- **Comprehensive**: 5-10 ms

### Optimization Tips

1. **Choose appropriate feature level**: Use minimal features for your task
2. **Disable unused options**: Set `include_descriptors=False` if not needed
3. **Batch processing**: Process multiple molecules together
4. **Memory management**: Clear large molecule lists from memory
5. **Preprocessing**: Validate SMILES before conversion

### Scaling Guidelines

| Dataset Size | Recommended Level | Additional Settings |
|-------------|------------------|-------------------|
| < 1K molecules | Any level | Full features |
| 1K - 10K | Standard/Extended | Consider disabling 3D |
| 10K - 100K | Basic/Standard | Disable partial charges |
| > 100K | Basic | Minimal features only |

## Troubleshooting

### Common Issues

#### SMILES Parsing Failures
```python
# Check for invalid SMILES
smiles = "invalid_smiles"
graph = converter.to_graph(smiles)
if graph is None:
    print(f"Failed to parse SMILES: {smiles}")
```

**Solutions**:
- Validate SMILES using RDKit's `Chem.MolFromSmiles()`
- Check for special characters or malformed structures
- Use canonical SMILES when possible

#### Memory Issues
```python
# Monitor memory usage
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb:.1f} MB")
```

**Solutions**:
- Reduce feature level
- Process in smaller batches  
- Disable unnecessary features
- Use garbage collection: `import gc; gc.collect()`

#### Descriptor Compatibility Issues
```python
# Check for missing descriptors
import warnings
from rdkit.Chem import rdMolDescriptors

# The converter handles these automatically, but you can check manually:
missing_descriptors = []
test_descriptors = ['BertzCT', 'Kappa1', 'Chi0v', 'HallKierAlpha']

for desc in test_descriptors:
    if not hasattr(rdMolDescriptors, desc):
        missing_descriptors.append(desc)

if missing_descriptors:
    print(f"Missing descriptors (fallbacks used): {missing_descriptors}")
    print("Consider upgrading RDKit: conda install -c conda-forge rdkit")
```

**Solutions**:
- The converter automatically uses chemically meaningful fallbacks
- Fallback values maintain the same feature dimensions
- No action required - converter handles this transparently

#### Version-Specific Behavior
```python
# Check current handling approach
converter = SMILESToGraph(feature_level="comprehensive")
atom_names, bond_names = converter.get_feature_names()

print("Current feature configuration:")
print(f"  Atom features: {len(atom_names)}")
print(f"  Bond features: {len(bond_names)}")

# Test with a molecule
graph = converter.to_graph("c1ccccc1")  # benzene
desc_available = len(graph['descriptors'])
print(f"  Molecular descriptors: {desc_available}")
```

#### Performance Issues
```python
import time

# Benchmark processing speed
start_time = time.time()
graphs = [converter.to_graph(s) for s in smiles_list[:100]]
end_time = time.time()

avg_time = (end_time - start_time) / 100
print(f"Average processing time: {avg_time*1000:.2f} ms per molecule")
```

**Solutions**:
- Reduce feature complexity
- Disable expensive calculations (partial charges, 3D)
- Consider parallel processing for large datasets

### Error Messages

#### "Invalid feature_level"
Check that feature level is one of: 'basic', 'standard', 'extended', 'comprehensive'

#### "Could not compute Gasteiger charges"
Some molecules may fail partial charge calculation. The converter continues with zero charges.

#### "Error calculating descriptors"
Some molecular descriptors may fail for unusual structures. Non-essential descriptors are skipped.

### Best Practices

1. **Validation**: Always check return values for None
2. **Consistency**: Use same converter configuration throughout pipeline
3. **Testing**: Validate feature dimensions match expected values
4. **Documentation**: Keep track of feature configurations used
5. **Monitoring**: Track processing failures and performance metrics

### Getting Help

For additional support:
- Check RDKit documentation for underlying chemistry
- Validate SMILES using chemical drawing software
- Test with simple molecules first
- Compare results across different feature levels
- Use descriptive error messages and logging

---

## Conclusion

This SMILES to Graph converter provides a flexible, chemistry-aware approach to molecular representation for machine learning. By choosing appropriate feature levels and configurations, you can optimize for your specific application while maintaining chemical interpretability and computational efficiency.

The fixed-size feature representations within each level ensure consistency for machine learning applications, while the comprehensive chemical grounding provides meaningful features for diverse molecular modeling tasks.
