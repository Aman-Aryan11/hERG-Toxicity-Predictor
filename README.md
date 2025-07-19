# hERG Toxicity Predictor - Web Deployment

This repository contains a Graph Neural Network (GNN) model for predicting hERG (human Ether-à-go-go-Related Gene) channel toxicity from molecular structures, deployed as a web application using Streamlit.

## 🧬 About hERG Toxicity

hERG channel inhibition is a major cause of drug-induced cardiac arrhythmias and is a critical safety concern in drug development. This model helps predict whether a molecule is likely to inhibit the hERG channel, which can lead to:

- Prolonged QT interval
- Cardiac arrhythmias  
- Sudden cardiac death

## 📊 Model Performance

- **ROC AUC**: 0.9139
- **Accuracy**: 0.8978

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd hERG_toxicity
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**
   Make sure `gnn_best_model.pt` is in the root directory of the project.

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the web interface**
   Open your browser and go to `http://localhost:8501`

## 🖥️ Web Interface Features

### Single Molecule Prediction
- Enter a SMILES string to get toxicity predictions
- View molecular structure visualization
- See confidence scores and risk levels

### Batch Processing
- Upload CSV files with SMILES column
- Process multiple molecules at once
- Download results as CSV

### Example Molecules
- Test with pre-loaded common molecules
- Includes aspirin, paracetamol, ibuprofen, caffeine, and nicotine

## 📁 File Structure

```
hERG_toxicity/
├── app.py                          # Main Streamlit application
├── gnn_best_model.pt              # Trained GNN model weights
├── requirements_deploy.txt         # Deployment dependencies
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── data/                          # Data files
│   ├── 41598_2019_47536_MOESM2_ESM.xlsx
│   └── output_file.csv
├── notebook/
│   └── model.ipynb                # Training notebook
└── README.md                      # This file
```

## 🔧 Model Architecture

The GNN model uses:
- **Molecular Graph Representation**: Converts SMILES to molecular graphs
- **NNConv Layers**: Message passing neural networks for graph convolution
- **Global Mean Pooling**: Aggregates node features to graph-level representation
- **Binary Classification Head**: Predicts toxic/non-toxic probability

### Input Features
- Atomic number
- Number of bonds (degree)
- Implicit valence
- Aromaticity
- Formal charge

### Bond Features
- Single bonds
- Double bonds
- Triple bonds
- Aromatic bonds

## 📊 Training Data

- **Total molecules**: 203,853
- **Non-toxic (Class 0)**: 147,235
- **Toxic (Class 1)**: 5,136
- **Data source**: 41598_2019_47536_MOESM2_ESM.xlsx

## 🔍 Usage Examples

### Single Prediction
```python
# Example SMILES strings
smiles_examples = [
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Aspirin
    "CC(=O)NC1=CC=C(O)C=C1",          # Paracetamol
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
]
```

### Batch Processing
Create a CSV file with format:
```csv
SMILES
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
CC(=O)NC1=CC=C(O)C=C1
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

## 🛠️ Troubleshooting

### Common Issues

1. **Model loading error**
   - Ensure `gnn_best_model.pt` exists in the project root
   - Check file permissions

2. **RDKit installation issues**
   - On Windows: `conda install -c conda-forge rdkit`
   - On Linux/Mac: `pip install rdkit`

3. **PyTorch Geometric issues**
   - Install PyTorch first: `pip install torch`
   - Then install PyG: `pip install torch-geometric`

4. **Memory issues**
   - Reduce batch size in batch processing
   - Use smaller model if available

### Performance Optimization

- Use GPU if available (CUDA-compatible)
- Increase batch size for faster processing
- Cache model loading with `@st.cache_resource`

## 📈 Model Interpretation

The model provides:
- **Binary classification**: Toxic (1) or Non-toxic (0)
- **Probability score**: Confidence in prediction (0-1)
- **Risk levels**: Low (<0.5), Medium (0.5-0.7), High (>0.7)

## 🔬 Research Context

This model was trained on a large dataset of molecules with known hERG inhibition data. The GNN architecture allows it to learn molecular patterns that correlate with hERG channel inhibition, providing valuable insights for drug safety assessment.

**Note**: This tool is for research purposes and should not be used as the sole basis for drug development decisions. Always consult with qualified professionals for drug safety assessment. # hERG-Toxicity-Predictor
