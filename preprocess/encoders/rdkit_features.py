"""RDKit features: canonical SMILES, ECFP4 (2048 bits), and 280-d physicochemical descriptors.

Descriptors are grouped into ten subsets following the MolBERT descriptor sets.
Within each subset, the values of a single compound are scaled among
themselves (RobustScaler followed by a Yeo-Johnson PowerTransformer), so a new
compound can be featurized without any reference to the training drugs.
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.preprocessing import PowerTransformer, RobustScaler

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_GROUPS = {
    "SURFACE": [
        "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5",
        "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "LabuteASA", "PEOE_VSA1", "PEOE_VSA10",
        "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5",
        "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3",
        "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10",
        "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6",
        "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "TPSA",
    ],
    "CHARGE": [
        "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge", "PEOE_VSA1",
        "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4",
        "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9",
    ],
    "FRAGMENT": [
        "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
        "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors",
        "NumHeteroatoms", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
        "NumSaturatedRings", "RingCount", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO",
        "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN",
        "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_Nhpyrrole",
        "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid", "fr_amide",
        "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur", "fr_benzene",
        "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether",
        "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole", "fr_imide",
        "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy",
        "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho", "fr_nitroso",
        "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond", "fr_phos_acid",
        "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide", "fr_prisulfonamd", "fr_pyridine",
        "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene", "fr_tetrazole",
        "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea",
    ],
    "SIMPLE": [
        "FpDensityMorgan2", "FractionCSP3", "MolLogP", "MolWt", "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
        "TPSA",
    ],
    "GRAPH": [
        "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n",
        "Chi3v", "Chi4n", "Chi4v", "HallKierAlpha", "Ipc", "Kappa1", "Kappa2", "Kappa3",
    ],
    "ESTATE": [
        "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5",
        "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "MaxAbsEStateIndex", "MaxEStateIndex",
        "MinAbsEStateIndex", "MinEStateIndex", "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3",
        "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9",
    ],
    "DRUGLIKENESS": [
        "ExactMolWt", "FractionCSP3", "HeavyAtomCount", "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount",
        "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles",
        "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
        "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings",
        "RingCount", "TPSA", "qed",
    ],
    "LOGP": [
        "MolLogP", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3",
        "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9",
    ],
    "REFRACTIVITY": [
        "MolMR", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7",
        "SMR_VSA8", "SMR_VSA9",
    ],
    "GENERAL": [
        "ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "HeavyAtomMolWt",
        "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge", "MolWt",
        "NumRadicalElectrons", "NumValenceElectrons",
    ],
}
# Subsets are concatenated in alphabetical order.
GROUP_ORDER = sorted(DESCRIPTOR_GROUPS)


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def ecfp4(smiles_list, n_bits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fps = [np.array(list(gen.GetFingerprint(Chem.MolFromSmiles(s))), dtype=np.int64) for s in smiles_list]
    return pd.DataFrame(np.stack(fps), columns=[f"ECFP4_{i}" for i in range(n_bits)])


def _scale_within_compound(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    x = RobustScaler().fit_transform(x)
    return PowerTransformer(method="yeo-johnson").fit_transform(x).ravel()


def descriptor_column_names():
    return [f"{group}_{name}" for group in GROUP_ORDER for name in DESCRIPTOR_GROUPS[group]]


def descriptors_280(smiles_list):
    rows = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        parts = []
        for group in GROUP_ORDER:
            raw = [getattr(Descriptors, name)(mol) for name in DESCRIPTOR_GROUPS[group]]
            parts.append(_scale_within_compound(raw))
        rows.append(np.concatenate(parts))
    return pd.DataFrame(np.stack(rows), columns=descriptor_column_names())
