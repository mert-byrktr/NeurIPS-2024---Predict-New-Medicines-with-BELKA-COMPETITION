## Encoding Molecular Structures

In this competition, the dataset provides molecular structures represented by SMILES strings. The goal was to encode these SMILES representations into numerical format, enabling them to be used as input features for machine learning models. We also incorporated chemical features associated with building blocks of the molecules to further enrich the dataset.

### 1. **SMILES Encoding**

To encode the SMILES strings into a usable format for machine learning models, we employed the following steps:

- **Custom Character Encoding**: Each character in the SMILES string was mapped to a unique integer using a custom dictionary. The length of the SMILES strings varied, so we padded all SMILES representations to a length of 142 characters for consistency.
  
  ```python
  enc = {'l': 1, 'y': 2, '@': 3, '3': 4, 'H': 5, 'S': 6, 'F': 7, 'C': 8, 'r': 9, 's': 10, ...}
  
  def encode_smile(smile):
      tmp = [enc[i] for i in smile]
      tmp = tmp + [0] * (142 - len(tmp))  # Padding
      return np.array(tmp).astype(np.uint8)
``

### 2. ** Binary Binding Labels**
For each protein target (BRD4, HSA, and sEH), the binding label was included in the dataset. These labels represent whether a molecule binds to the specific protein target.
