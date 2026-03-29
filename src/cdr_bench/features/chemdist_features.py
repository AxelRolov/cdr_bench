import pandas as pd
import torch
from typing import Dict, Any
from dgl import batch, DGLGraph
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from src.cdr_bench.io_utils.data_preprocessing import remove_duplicates  # NOTE: remove_duplicate_rows was removed; callers need updating

# Atom and Bond Featurizers
NF = CanonicalAtomFeaturizer()
BF = CanonicalBondFeaturizer()


# Custom function definitions


def chemdist_func(smiles, model, NF, BF):
    """
    Process a single SMILES string into a molecular graph and generate an embedding using the provided model.
    """
    g = smiles_to_bigraph(smiles=smiles, node_featurizer=NF, edge_featurizer=BF)
    nfeats = g.ndata.pop('h')
    efeats = g.edata.pop('e')

    with torch.no_grad():
        output = model._net(g, nfeats, efeats).detach().numpy()

    return output


def chemdist_func_batch(graph_list, model, NF, BF):
    """
    Process a batch of molecular graphs and generate embeddings using the provided model.
    """
    bg = batch(graph_list)
    bg = bg.to('cuda')
    nfeats = bg.ndata.pop('h')
    efeats = bg.edata.pop('e')

    with torch.no_grad():
        output = model._net(bg, nfeats, efeats).cpu().detach().numpy()

    del bg
    del nfeats
    del efeats
    torch.cuda.empty_cache()

    return output

# Initialize Atom and Bond Featurizers
NF = CanonicalAtomFeaturizer()
BF = CanonicalBondFeaturizer()

def load_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Loads and initializes the model with the specified parameters from the configuration.

    Args:
        config (Dict[str, Any]): Dictionary with model configuration, including model path and parameters.

    Returns:
        torch.nn.Module: The initialized and loaded model, ready for evaluation.
    """
    model = DistanceNetworkLigthning(**config["model_params"])

    state_dict = torch.load(config["model_path"])
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)

    model.eval()
    model.cuda()
    return model


def generate_embeddings(df: pd.DataFrame, model: torch.nn.Module,
                        node_featurizer: CanonicalAtomFeaturizer,
                        edge_featurizer: CanonicalBondFeaturizer) -> pd.DataFrame:
    """
    Generates graph-based embeddings for molecules in the DataFrame, stores them in a new column,
    and removes rows with NaN embeddings.

    Args:
        df (pd.DataFrame): DataFrame containing a 'smi' column with SMILES strings.
        model (torch.nn.Module): Pre-trained model to generate embeddings.
        node_featurizer (CanonicalAtomFeaturizer): Featurizer for atoms in the molecular graph.
        edge_featurizer (CanonicalBondFeaturizer): Featurizer for bonds in the molecular graph.

    Returns:
        pd.DataFrame: Updated DataFrame with a new 'embed' column containing embeddings, and rows
                      with NaN values in the 'embed' column removed.
    """
    # Convert SMILES strings to molecular graphs
    graph_list = df['smi'].apply(lambda x: smiles_to_bigraph(smiles=x,
                                                             node_featurizer=node_featurizer,
                                                             edge_featurizer=edge_featurizer))

    # Generate embeddings in batches
    embeddings = chemdist_func_batch(graph_list.tolist(), model, node_featurizer, edge_featurizer)

    # Add embeddings to DataFrame and remove rows with NaN embeddings
    df['embed'] = pd.Series(list(embeddings))
    df = df.dropna(subset=['embed'])
    df = remove_duplicate_rows(df, 'embed')

    return df