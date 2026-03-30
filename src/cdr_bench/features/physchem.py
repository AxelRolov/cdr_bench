from rdkit import Chem
from rdkit.Chem import Descriptors


def calculate_descriptors(mol: Chem.Mol) -> list[float | None]:
    """
    Calculate six physicochemical properties: HBD, HBA, LogP, MW, TPSA, and RTB.

    Args:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        List[Optional[float]]: List of calculated properties or None for invalid molecules.
    """
    if mol:
        hbd = Descriptors.NumHDonors(mol)  # Hydrogen bond donors
        hba = Descriptors.NumHAcceptors(mol)  # Hydrogen bond acceptors
        logp = Descriptors.MolLogP(mol)  # LogP
        mw = Descriptors.MolWt(mol)  # Molecular weight
        tpsa = Descriptors.TPSA(mol)  # Topological polar surface area
        rtb = Descriptors.NumRotatableBonds(mol)  # Rotatable bonds
        return [hbd, hba, logp, mw, tpsa, rtb]
    else:
        return [None] * 6
