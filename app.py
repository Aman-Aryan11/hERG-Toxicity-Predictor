import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="hERG Toxicity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .toxic {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .non-toxic {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# GNN Model Definition (same as in notebook)
class GNNBinaryClassifier(nn.Module):
    def __init__(self, node_input_dim=5, edge_input_dim=1, hidden_dim=64):
        super(GNNBinaryClassifier, self).__init__()

        # Edge MLP 1 → for NNConv layer 1
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim * node_input_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * node_input_dim, hidden_dim * node_input_dim)
        )

        self.conv1 = NNConv(
            in_channels=node_input_dim,
            out_channels=hidden_dim,
            nn=self.edge_mlp1,
            aggr='mean'
        )
        self.norm1 = BatchNorm(hidden_dim)

        # Edge MLP 2 → for NNConv layer 2
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
        )

        self.conv2 = NNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            nn=self.edge_mlp2,
            aggr='mean'
        )
        self.norm2 = BatchNorm(hidden_dim)

        # Final MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.relu(x)

        # Graph-level representation
        pooled = global_mean_pool(x, batch)

        # Output logits
        out = self.classifier(pooled)
        return out

# SMILES to Graph conversion functions
def get_atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),                # Atomic number
        atom.GetDegree(),                   # Number of bonds
        atom.GetImplicitValence(),          # Implicit valence
        int(atom.GetIsAromatic()),          # Aromaticity
        atom.GetFormalCharge(),             # Formal charge
    ], dtype=torch.float)

# Bond type mapping
bond_type_to_int = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Atom features
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        if not atom_features:
            return None
        x = torch.stack(atom_features)

        # Edge index and edge attributes
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_encoding = bond_type_to_int.get(bond_type, -1)
            if bond_encoding == -1:
                continue  # skip unknown bond types

            # Add edges in both directions
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([[bond_encoding], [bond_encoding]])

        # Convert to tensors
        if not edge_index:
            return None  # skip molecules with no valid bonds

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    except Exception as e:
        return None

@st.cache_resource
def load_model():
    """Load the trained GNN model"""
    try:
        model = GNNBinaryClassifier()
        model.load_state_dict(torch.load("gnn_best_model.pt", map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_toxicity(model, smiles):
    """Predict toxicity for a given SMILES string"""
    try:
        # Convert SMILES to graph
        graph = smiles_to_graph(smiles)
        if graph is None:
            return None, "Invalid SMILES string"
        
        # Add batch dimension
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        
        # Make prediction
        with torch.no_grad():
            output = model(graph)
            probability = torch.sigmoid(output).item()
        
        return probability, None
    except Exception as e:
        return None, f"Prediction error: {e}"

def plot_molecule(smiles):
    """Generate molecular structure visualization"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Use RDKit's built-in drawing
        from rdkit.Chem import Draw
        img = Draw.MolToImage(mol, size=(300, 300))
        
        # Convert to base64 for display
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    except Exception as e:
        st.error(f"Error generating molecule visualization: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 hERG Toxicity Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    This application uses a Graph Neural Network (GNN) to predict hERG (human Ether-à-go-go-Related Gene) 
    channel toxicity from molecular structures. Enter a SMILES string to get toxicity predictions.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if 'gnn_best_model.pt' exists in the current directory.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Model Information")
    st.sidebar.markdown("""
    **Model Performance:**
    - ROC AUC: 0.9139
    - Accuracy: 0.8978
    """)
    
    st.sidebar.markdown("""
    **About hERG Toxicity:**
    hERG channel inhibition is a major cause of drug-induced cardiac arrhythmias and 
    is a critical safety concern in drug development.
    """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔬 Molecular Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Single SMILES", "Batch Upload", "Example Molecules"]
        )
        
        if input_method == "Single SMILES":
            smiles_input = st.text_input(
                "Enter SMILES string:",
                placeholder="e.g., CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                help="Enter a valid SMILES string representing a chemical structure"
            )
            
            if st.button("Predict Toxicity", type="primary"):
                if smiles_input.strip():
                    with st.spinner("Processing..."):
                        probability, error = predict_toxicity(model, smiles_input.strip())
                        
                        if error:
                            st.error(error)
                        else:
                            # Display results
                            st.subheader("📋 Prediction Results")
                            
                            # Create prediction box
                            is_toxic = probability > 0.5
                            box_class = "toxic" if is_toxic else "non-toxic"
                            status = "🚨 TOXIC" if is_toxic else "✅ NON-TOXIC"
                            
                            st.markdown(f"""
                            <div class="prediction-box {box_class}">
                                <h3>{status}</h3>
                                <p><strong>Confidence:</strong> {probability:.1%}</p>
                                <p><strong>Risk Level:</strong> {'High' if probability > 0.7 else 'Medium' if probability > 0.5 else 'Low'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display molecule structure
                            img_str = plot_molecule(smiles_input.strip())
                            if img_str:
                                st.subheader("🧪 Molecular Structure")
                                st.image(f"data:image/png;base64,{img_str}", width=300)
                            
                            # Confidence gauge
                            st.subheader("📊 Confidence Score")
                            st.progress(probability)
                            st.caption(f"Toxicity probability: {probability:.1%}")
                            
        elif input_method == "Batch Upload":
            st.subheader("📁 Batch Processing")
            uploaded_file = st.file_uploader(
                "Upload CSV file with SMILES column",
                type=['csv'],
                help="CSV file should contain a column named 'SMILES' with molecular structures"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'SMILES' not in df.columns:
                        st.error("CSV file must contain a 'SMILES' column")
                    else:
                        st.success(f"Loaded {len(df)} molecules")
                        
                        if st.button("Process Batch", type="primary"):
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, row in df.iterrows():
                                probability, error = predict_toxicity(model, row['SMILES'])
                                results.append({
                                    'SMILES': row['SMILES'],
                                    'Toxicity_Probability': probability if probability is not None else 'Error',
                                    'Prediction': 'Toxic' if probability and probability > 0.5 else 'Non-toxic' if probability else 'Error',
                                    'Error': error
                                })
                                progress_bar.progress((i + 1) / len(df))
                            
                            # Display results
                            results_df = pd.DataFrame(results)
                            st.subheader("📊 Batch Results")
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name="toxicity_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            valid_results = results_df[results_df['Error'].isna()]
                            if len(valid_results) > 0:
                                toxic_count = len(valid_results[valid_results['Prediction'] == 'Toxic'])
                                non_toxic_count = len(valid_results[valid_results['Prediction'] == 'Non-toxic'])
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Molecules", len(df))
                                with col2:
                                    st.metric("Toxic", toxic_count)
                                with col3:
                                    st.metric("Non-toxic", non_toxic_count)
                
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        
        else:  # Example Molecules
            st.subheader("🧪 Example Molecules")
            
            examples = {
                "Aspirin": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
                "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "Nicotine": "CN1CCC[C@H]1C2=CN=CC=C2"
            }
            
            selected_example = st.selectbox("Choose an example molecule:", list(examples.keys()))
            example_smiles = examples[selected_example]
            
            st.text_input("SMILES:", value=example_smiles, key="example_smiles")
            
            if st.button("Predict for Example", type="primary"):
                with st.spinner("Processing..."):
                    probability, error = predict_toxicity(model, example_smiles)
                    
                    if error:
                        st.error(error)
                    else:
                        st.subheader("📋 Prediction Results")
                        
                        is_toxic = probability > 0.5
                        box_class = "toxic" if is_toxic else "non-toxic"
                        status = "🚨 TOXIC" if is_toxic else "✅ NON-TOXIC"
                        
                        st.markdown(f"""
                        <div class="prediction-box {box_class}">
                            <h3>{status}</h3>
                            <p><strong>Confidence:</strong> {probability:.1%}</p>
                            <p><strong>Risk Level:</strong> {'High' if probability > 0.7 else 'Medium' if probability > 0.5 else 'Low'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display molecule structure
                        img_str = plot_molecule(example_smiles)
                        if img_str:
                            st.subheader("🧪 Molecular Structure")
                            st.image(f"data:image/png;base64,{img_str}", width=300)
    
    with col2:
        st.subheader("ℹ️ Information")
        st.markdown("""
        **What is hERG Toxicity?**
        
        hERG (human Ether-à-go-go-Related Gene) encodes a potassium channel that plays a crucial role in cardiac repolarization. Inhibition of this channel can lead to:
        
        - Prolonged QT interval
        - Cardiac arrhythmias
        - Sudden cardiac death
        
        **How to use:**
        1. Enter a SMILES string or upload a CSV file
        2. Click "Predict Toxicity"
        3. View the prediction results and confidence score
        
        **SMILES Format:**
        SMILES (Simplified Molecular Input Line Entry System) is a notation for describing molecular structures using ASCII characters.
        """)
        
        st.subheader("📈 Model Details")
        st.markdown("""
        **Architecture:** Graph Neural Network (GNN)
        - Uses molecular graph representation
        - NNConv layers for message passing
        - Global mean pooling for graph-level prediction
        
        **Training Data:** 203,853 molecules
        - Class 0 (Non-toxic): 147,235
        - Class 1 (Toxic): 5,136
        """)

if __name__ == "__main__":
    main() 