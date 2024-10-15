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

### 2. Binary Binding Labels
For each protein target (BRD4, HSA, and sEH), the binding label was included in the dataset. These labels represent whether a molecule binds to the specific protein target.

### 3. Building Block Features

Molecules in the dataset are composed of three main building blocks. We extracted additional features for these building blocks from external chemical property datasets:

Merging Physicochemical Features: Using the building block SMILES strings (buildingblock3_smiles), we merged external physicochemical features into the dataset. These features include molecular weight, hydrophobicity, and other important chemical properties.

## Models

In this competition, we experimented with three different deep learning architectures: **1D CNN**, **LSTM**, and **Transformer**. Each of these models was designed to predict the binary binding classifications of small molecules to three protein targets (BRD4, HSA, and sEH) based on SMILES representations. Below is a summary of the models and their respective configurations.

### 1. **1D Convolutional Neural Network (1D CNN)**

The 1D CNN model leverages convolutional layers to capture local patterns in the encoded SMILES strings. We used five convolutional layers with increasing numbers of filters, followed by global max pooling and fully connected dense layers to make predictions.

- **Architecture**:
  - Embedding layer with 36 inputs and 128-dimensional embeddings.
  - Five convolutional layers with filters increasing from 32 to 160.
  - Global max pooling followed by fully connected dense layers with dropout for regularization.
  - Final output layer with 3 neurons and sigmoid activation to predict the binding probabilities for each protein target.

- **Key Parameters**:
  - Batch size: 4096
  - Learning rate: 1e-3
  - Weight decay: 0.05

```python
x = tf.keras.layers.Conv1D(filters=NUM_FILTERS, kernel_size=3, activation='relu', padding='valid', strides=1)(x)
x = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(3, activation='sigmoid')(x)
```

### 2. LSTM (Long Short-Term Memory)
The LSTM model is designed to capture sequential dependencies in the SMILES representation. We used three LSTM layers to process the encoded sequences, followed by dense layers for final prediction.

Architecture:

Embedding layer with 36 inputs and 128-dimensional embeddings.
Three LSTM layers with 128, 64, and 32 units respectively.
Fully connected dense layers with dropout for regularization.
Final output layer with 3 neurons and sigmoid activation.
Key Parameters:

Batch size: 4096
Learning rate: 1e-3
Weight decay: 0.05

```python
x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='sigmoid')(x)

```

### 3. Transformer
The Transformer model is designed to capture long-range dependencies in the SMILES strings. This model is adapted from similar competitions' winning solutions.
Architecture:

Embedding layer with 36 inputs and 128-dimensional embeddings.
Positional encoding to maintain the order of the sequence.
Four Transformer blocks, each containing a multi-head attention layer followed by a feed-forward network.
Global average pooling followed by fully connected dense layers.
Final output layer with 3 neurons and sigmoid activation.
Key Parameters:

Batch size: 4096
Learning rate: 1e-3
Weight decay: 0.05

```python
attn_output = self.att(inputs, inputs, attention_mask=att_mask)
out1 = self.layernorm1(inputs + attn_output)
ffn_output = self.ffn(out1)
ffn_output = self.dropout2(ffn_output, training=training)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = tf.keras.layers.Dense(3, activation='sigmoid')(x)

```

### 4. Training Strategy
We used the TPU and GPU for faster training and ensured the setup would adapt based on available hardware. The model was trained using a stratified k-fold approach to handle the imbalanced dataset, with 15 folds and selected folds used for training.

Folds: We employed 15 folds with selected folds [0, 2, 4, 6] used in the final training.
