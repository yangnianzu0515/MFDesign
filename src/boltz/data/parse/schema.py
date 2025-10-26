from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import click
import numpy as np
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Conformer, Mol

from boltz.data import const
from boltz.data.types import (
    Atom,
    Bond,
    Chain,
    ChainInfo,
    Connection,
    Interface,
    InferenceOptions,
    Record,
    Residue,
    Structure,
    AntibodyInfo,
    Target,
)

####################################################################################################
# DATACLASSES
####################################################################################################


@dataclass(frozen=True)
class ParsedAtom:
    """A parsed atom object."""

    name: str
    element: int
    charge: int
    coords: tuple[float, float, float]
    conformer: tuple[float, float, float]
    is_present: bool
    chirality: int
    is_spec_atom: bool


@dataclass(frozen=True)
class ParsedBond:
    """A parsed bond object."""

    atom_1: int
    atom_2: int
    type: int


@dataclass(frozen=True)
class ParsedResidue:
    """A parsed residue object."""

    name: str
    type: int
    idx: int
    atoms: list[ParsedAtom]
    bonds: list[ParsedBond]
    orig_idx: Optional[int]
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool
    is_spec_residue: bool


@dataclass(frozen=True)
class ParsedChain:
    """A parsed chain object."""

    entity: str
    type: str
    residues: list[ParsedResidue]
    # residues: a list of ParsedResidue objects


####################################################################################################
# HELPERS
####################################################################################################


def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    Tuple[int, int, int, int]
        The converted atom name.

    """
    name = name.strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)


def compute_3d_conformer(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coordinates using EKTDG method.

    Taken from `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The RDKit molecule to process
    version: str, optional
        The ETKDG version, defaults ot v3

    Returns
    -------
    bool
        Whether computation was successful.

    """
    if version == "v3":
        options = AllChem.ETKDGv3()
    elif version == "v2":
        options = AllChem.ETKDGv2()
    else:
        options = AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1

    try:
        conf_id = AllChem.EmbedMolecule(mol, options)

        if conf_id == -1:
            print(f"WARNING: RDKit ETKDGv3 failed to generate a conformer for molecule "
                  f"{Chem.MolToSmiles(AllChem.RemoveHs(mol))}, so the program will start with random coordinates. "
                  f"Note that the performance of the model under this behaviour was not tested.")
            options.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, options)

        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    except RuntimeError:
        pass  # Force field issue here
    except ValueError:
        pass  # sanitization issue here

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", "Computed")
        conformer.SetProp("coord_generation", f"ETKDG{version}")

        return True

    return False


def get_conformer(mol: Mol) -> Conformer:
    """Retrieve an rdkit object for a deemed conformer.

    Inspired by `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The molecule to process.

    Returns
    -------
    Conformer
        The desired conformer, if any.

    Raises
    ------
    ValueError
        If there are no conformers of the given tyoe.

    """
    # Try using the computed conformer
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Computed":
                return c
        except KeyError:  # noqa: PERF203
            pass

    # Fallback to the ideal coordinates
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Ideal":
                return c
        except KeyError:  # noqa: PERF203
            pass

    msg = "Conformer does not exist."
    raise ValueError(msg)


####################################################################################################
# PARSING
####################################################################################################


def parse_ccd_residue(
    name: str,
    ref_mol: Mol,
    res_idx: int,
    is_spec: bool,
) -> Optional[ParsedResidue]:
    """Parse an MMCIF ligand.

    First tries to get the SMILES string from the RCSB.
    Then, tries to infer atom ordering using RDKit.

    Parameters
    ----------
    name: str
        The name of the molecule to parse.
    ref_mol: Mol
        The reference molecule to parse.
    res_idx : int
        The residue index.

    Returns
    -------
    ParsedResidue, optional
       The output ParsedResidue, if successful.

    """
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    # Remove hydrogens
    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)

    # Check if this is a single atom CCD residue
    if ref_mol.GetNumAtoms() == 1:
        pos = (0, 0, 0)
        ref_atom = ref_mol.GetAtoms()[0]
        chirality_type = const.chirality_type_ids.get(
            ref_atom.GetChiralTag(), unk_chirality
        )
        atom = ParsedAtom(
            name=ref_atom.GetProp("name"),
            element=ref_atom.GetAtomicNum(),
            charge=ref_atom.GetFormalCharge(),
            coords=pos,
            conformer=(0, 0, 0),
            is_present=True,
            chirality=chirality_type,
            is_spec_atom=is_spec,
        )
        unk_prot_id = const.unk_token_ids["PROTEIN"]
        residue = ParsedResidue(
            name=name,
            type=unk_prot_id,
            atoms=[atom],
            bonds=[],
            idx=res_idx,
            orig_idx=None,
            atom_center=0,  # Placeholder, no center
            atom_disto=0,  # Placeholder, no center
            is_standard=False,
            is_present=True,
            is_spec_residue=is_spec,
        )
        return residue

    # Get reference conformer coordinates
    conformer = get_conformer(ref_mol)

    # Parse each atom in order of the reference mol
    atoms = []
    atom_idx = 0
    idx_map = {}  # Used for bonds later

    for i, atom in enumerate(ref_mol.GetAtoms()):
        # Get atom name, charge, element and reference coordinates
        atom_name = atom.GetProp("name")
        charge = atom.GetFormalCharge()
        element = atom.GetAtomicNum()
        ref_coords = conformer.GetAtomPosition(atom.GetIdx())
        ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)
        chirality_type = const.chirality_type_ids.get(
            atom.GetChiralTag(), unk_chirality
        )

        # Get PDB coordinates, if any
        coords = (0, 0, 0)
        atom_is_present = True

        # Add atom to list
        atoms.append(
            ParsedAtom(
                name=atom_name,
                element=element,
                charge=charge,
                coords=coords,
                conformer=ref_coords,
                is_present=atom_is_present,
                chirality=chirality_type,
                is_spec_atom=is_spec,
            )
        )
        idx_map[i] = atom_idx
        atom_idx += 1  # noqa: SIM113

    # Load bonds
    bonds = []
    unk_bond = const.bond_type_ids[const.unk_bond_type]
    for bond in ref_mol.GetBonds():
        idx_1 = bond.GetBeginAtomIdx()
        idx_2 = bond.GetEndAtomIdx()

        # Skip bonds with atoms ignored
        if (idx_1 not in idx_map) or (idx_2 not in idx_map):
            continue

        idx_1 = idx_map[idx_1]
        idx_2 = idx_map[idx_2]
        start = min(idx_1, idx_2)
        end = max(idx_1, idx_2)
        bond_type = bond.GetBondType().name
        bond_type = const.bond_type_ids.get(bond_type, unk_bond)
        bonds.append(ParsedBond(start, end, bond_type))

    unk_prot_id = const.unk_token_ids["PROTEIN"]
    return ParsedResidue(
        name=name,
        type=unk_prot_id,
        atoms=atoms,
        bonds=bonds,
        idx=res_idx,
        atom_center=0,
        atom_disto=0,
        orig_idx=None,
        is_standard=False,
        is_present=True,
        is_spec_residue=is_spec,
    )


def parse_polymer(
    sequence: list[str],
    entity: str,
    chain_type: str,
    # chain_type always is 0 in our ab design
    components: dict[str, Mol],
    mask: np.ndarray,
) -> Optional[ParsedChain]:
    """Process a sequence into a chain object.

    Performs alignment of the full sequence to the polymer
    residues. Loads coordinates and masks for the atoms in
    the polymer, following the ordering in const.atom_order.

    Parameters
    ----------
    sequence : list[str]
        The full sequence of the polymer.
    entity : str
        The entity id.
    entity_type : str
        The entity type.
    components : dict[str, Mol]
        The preprocessed PDB components dictionary.

    Returns
    -------
    ParsedChain, optional
        The output chain, if successful.

    Raises
    ------
    ValueError
        If the alignment fails.

    """
    ref_res = set(const.tokens)
    # ref_res is the set of all tokens
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]
    # for protein, unk_chirality is CHI_UNSPECIFIED


    # Get coordinates and masks
    parsed = []
    for res_idx, res_name in enumerate(sequence):
        # Check if modified residue
        # Map MSE to MET
        res_corrected = res_name if res_name != "MSE" else "MET"


        # we don't support modifications for now in our ab design, so we can skip this
        # Handle non-standard residues
        if res_corrected not in ref_res: # there exist modifications in the input protein sequence
            ref_mol = components[res_corrected]
            residue = parse_ccd_residue(
                name=res_corrected,
                ref_mol=ref_mol,
                res_idx=res_idx,
                is_spec=mask[res_idx],
            )
            parsed.append(residue)
            continue


        # the ccd.pkl includes residues besides the ligands
        # Load ref residue
        ref_mol = components[res_corrected]
        # ref_mol is a rdkit mol object
        ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
        # remove hydrogens
        ref_conformer = get_conformer(ref_mol)
        # obtain the reference conformer for the residue

        # Only use reference atoms set in constants
        ref_name_to_atom = {a.GetProp("name"): a for a in ref_mol.GetAtoms()}
        # ref_name_to_atom is a dictionary, key is the atom name, value is the atom object
        ref_atoms = [ref_name_to_atom[a] for a in const.ref_atoms[res_corrected]]
        # ref_atoms is a list of atoms info (value in the ref_name_to_atom dictionary)

        # Iterate, always in the same order
        atoms: list[ParsedAtom] = []

        for ref_atom in ref_atoms:
            # Get atom name
            atom_name = ref_atom.GetProp("name") 
            # Get the atom index
            idx = ref_atom.GetIdx()

            # Get conformer coordinates
            ref_coords = ref_conformer.GetAtomPosition(idx)
            ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)
            # a tuple of coordinates for each atom
            
            
            # Set 0 coordinate
            atom_is_present = True
            coords = (0, 0, 0)

            # Add atom to list
            atoms.append(
                ParsedAtom(
                    name=atom_name,
                    element=ref_atom.GetAtomicNum(),
                    # element is the atomic number of the atom
                    charge=ref_atom.GetFormalCharge(),
                    # charge is the formal charge of the atom
                    coords=coords,
                    # coords is the coordinates of the atom
                    conformer=ref_coords,
                    # conformer is the reference coordinates of the atom
                    is_present=atom_is_present,
                    # is_present is whether the atom is present
                    chirality=const.chirality_type_ids.get(
                        ref_atom.GetChiralTag(), unk_chirality
                    ),
                    is_spec_atom=mask[res_idx],
                )
            )

        atom_center = const.res_to_center_atom_id[res_corrected]
        # atom_center is the index of the center atom of the residue. For residues, it is the CA atom
        atom_disto = const.res_to_disto_atom_id[res_corrected]
        # atom_disto is the index of the distal atom of the residue. For residues, it is the CB atom
        parsed.append(
            ParsedResidue(
                name=res_corrected,
                # name is the residue name
                type=const.token_ids[res_corrected],
                # id number of the residue
                atoms=atoms,
                # atoms: a list of atoms info
                bonds=[],
                idx=res_idx,
                atom_center=atom_center,
                atom_disto=atom_disto,
                is_standard=True,
                is_present=True,
                orig_idx=None,
                is_spec_residue=mask[res_idx]
            )
        )

    # Return polymer object
    return ParsedChain(
        entity=entity,
        residues=parsed,
        type=chain_type,
    )


# This function is used to parse the input file like yaml
def parse_boltz_schema(  # noqa: C901, PLR0915, PLR0912
    name: str,
    schema: dict,
    ccd: Mapping[str, Mol],
) -> tuple[Target, str]:
    """Parse a Boltz input yaml / json.

    The input file should be a dictionary with the following format:

    version: 1
    sequences:
        - protein:
            id: A
            sequence: "MADQLTEEQIAEFKEAFSLF"
            msa: path/to/msa1.a3m
        - protein:
            id: [B, C]
            sequence: "AKLSILPWGHC"
            msa: path/to/msa2.a3m
        - rna:
            id: D
            sequence: "GCAUAGC"
        - ligand:
            id: E
            smiles: "CC1=CC=CC=C1"
        - ligand:
            id: [F, G]
            ccd: []
    constraints:
        - bond:
            atom1: [A, 1, CA]
            atom2: [A, 2, N]
        - pocket:
            binder: E
            contacts: [[B, 1], [B, 2]]

    Parameters
    ----------
    name : str
        A name for the input.
    schema : dict
        The input schema.
    components : dict
        Dictionary of CCD components.

    Returns
    -------
    Target
        The parsed target.

    """
    # Assert version 1
    version = schema.get("version", 1)
    if version != 1:
        msg = f"Invalid version {version} in input!"
        raise ValueError(msg)

    # Disable rdkit warnings
    blocker = rdBase.BlockLogs()  # noqa: F841

    # First group items that have the same type, sequence and modifications
    items_to_group = {}
    for item in schema["sequences"]:
        # Get entity type
        entity_type = next(iter(item.keys())).lower()
        # Check if the entity type is valid
        if entity_type not in {"protein", "dna", "rna", "ligand"}:
            msg = f"Invalid entity type: {entity_type}"
            raise ValueError(msg)

        # Get sequence
        if entity_type in {"protein", "dna", "rna"}:
            seq = str(item[entity_type]["sequence"])
        # we don't accept ligand as input for now in our ab design
        elif entity_type == "ligand":
            assert "smiles" in item[entity_type] or "ccd" in item[entity_type]
            assert "smiles" not in item[entity_type] or "ccd" not in item[entity_type]
            if "smiles" in item[entity_type]:
                seq = str(item[entity_type]["smiles"])
            else:
                seq = str(item[entity_type]["ccd"])
        items_to_group.setdefault((entity_type, seq), []).append(item)


    # Go through entities and parse them
    seq_gt: str = ""
    spec_masks: str = ""
    chains: dict[str, ParsedChain] = {}
    chain_to_msa: dict[str, str] = {}
    entity_to_seq: dict[str, str] = {}
    entity_to_gt: dict[str, str] = {}
    entity_to_spec_mask: dict[str, str] = {}
    is_msa_custom = False
    is_msa_auto = False
    for entity_id, items in enumerate(items_to_group.values()):
        # Get entity type and sequence
        entity_type = next(iter(items[0].keys())).lower()

        # Ensure all the items share the same msa
        msa = -1
        if entity_type == "protein":
            # Get the msa, default to 0, meaning auto-generated
            msa = items[0][entity_type].get("msa", 0)
            if (msa is None) or (msa == ""):
                msa = 0
            # msa == 0 means we need to generate the msa


            # Check if all MSAs are the same within the same entity
            for item in items:
                item_msa = item[entity_type].get("msa", 0)
                if (item_msa is None) or (item_msa == ""):
                    item_msa = 0

                if item_msa != msa:
                    msg = "All proteins with the same sequence must share the same MSA!"
                    raise ValueError(msg)

            # Set the MSA, warn if passed in single-sequence mode
            if msa == "empty":
                msa = -1
                msg = (
                    "Found explicit empty MSA for some proteins, will run "
                    "these in single sequence mode. Keep in mind that the "
                    "model predictions will be suboptimal without an MSA."
                )
                click.echo(msg)

            if msa not in (0, -1):
                is_msa_custom = True
            elif msa == 0:
                is_msa_auto = True


        # we only consider protein type for now in out ab design
        # Parse a polymer
        if entity_type in {"protein", "dna", "rna"}:
            # Get token map
            if entity_type == "rna":
                token_map = const.rna_letter_to_token
            elif entity_type == "dna":
                token_map = const.dna_letter_to_token
            # we only consider protein type for now in out ab design
            elif entity_type == "protein":
                token_map = const.prot_letter_to_token
            else:
                msg = f"Unknown polymer type: {entity_type}"
                raise ValueError(msg)

            # Get polymer info 
            chain_type = const.chain_type_ids[entity_type.upper()] 
            # protein's chain type id is 0
            unk_token = const.unk_token[entity_type.upper()]
            # for unknown residue in protein, we use UNK as the token
            

            # Extract sequence
            seq = items[0][entity_type]["sequence"]
            entity_to_seq[entity_id] = seq
            
            # Convert sequence to tokens
            seq = [token_map.get(c, unk_token) for c in list(seq)]
            if "spec_mask" in items[0][entity_type]:
                spec_mask = items[0][entity_type]["spec_mask"]
                entity_to_spec_mask[entity_id] = spec_mask
                spec_mask = np.array(list(spec_mask), dtype=int).astype(bool)
            else:
                spec_mask = np.array(seq) == unk_token
                entity_to_spec_mask[entity_id] = "".join(spec_mask.astype(int).astype(str))

            assert len(seq) == len(spec_mask)

            if "ground_truth" in items[0][entity_type]:
                gt_seq = items[0][entity_type]["ground_truth"]
                assert len(seq) == len(gt_seq)
            else:
                gt_seq = ""
            
            entity_to_gt[entity_id] = gt_seq

            # we don't support modifications for now in our ab design, so we can skip this
            # Apply modifications
            for mod in items[0][entity_type].get("modifications", []):
                code = mod["ccd"]
                idx = mod["position"] - 1  # 1-indexed
                seq[idx] = code

            # Parse a polymer
            parsed_chain = parse_polymer(
                sequence=seq,
                entity=entity_id,
                chain_type=chain_type, 
                # chain type always is 0 in our ab design
                components=ccd,
                mask=spec_mask
            )


        # we don't support ligand as input for now in our ab design, so we can skip this
        # Parse a non-polymer
        elif (entity_type == "ligand") and "ccd" in (items[0][entity_type]):
            seq = items[0][entity_type]["ccd"]
            if isinstance(seq, str):
                seq = [seq]

            residues = []
            for code in seq:
                if code not in ccd:
                    msg = f"CCD component {code} not found!"
                    raise ValueError(msg)

                # Parse residue
                residue = parse_ccd_residue(
                    name=code,
                    ref_mol=ccd[code],
                    res_idx=0,
                    is_spec=False,
                )
                residues.append(residue)

            # Create multi ligand chain
            parsed_chain = ParsedChain(
                entity=entity_id,
                residues=residues,
                type=const.chain_type_ids["NONPOLYMER"],
            )
        elif (entity_type == "ligand") and ("smiles" in items[0][entity_type]):
            seq = items[0][entity_type]["smiles"]
            mol = AllChem.MolFromSmiles(seq)
            mol = AllChem.AddHs(mol)

            # Set atom names
            canonical_order = AllChem.CanonicalRankAtoms(mol)
            for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
                atom.SetProp("name", atom.GetSymbol().upper() + str(can_idx + 1))

            success = compute_3d_conformer(mol)
            if not success:
                msg = f"Failed to compute 3D conformer for {seq}"
                raise ValueError(msg)

            mol_no_h = AllChem.RemoveHs(mol)
            residue = parse_ccd_residue(
                name="LIG",
                ref_mol=mol_no_h,
                res_idx=0,
                is_spec=False,
            )
            parsed_chain = ParsedChain(
                entity=entity_id,
                residues=[residue],
                type=const.chain_type_ids["NONPOLYMER"],
            )
        else:
            msg = f"Invalid entity type: {entity_type}"
            raise ValueError(msg)

        # Add as many chains as provided ids
        for item in items:
            ids = item[entity_type]["id"]
            if isinstance(ids, str):
                ids = [ids]
            for chain_name in ids:
                chains[chain_name] = parsed_chain
                chain_to_msa[chain_name] = msa
    
    """
    chains: a dictionary of parsed chains
    key: chain name, e.g. A, B, C, D, E, F, G
    value: ParsedChain object
    
    if we have precomputed msa, the value for each key in chain_to_msa is the precomputeed msa file path;
    if we have not precomputed msa, the value for each key in chain_to_msa is 0
    """
                

    # Check if msa is custom or auto
    if is_msa_custom and is_msa_auto:
        msg = "Cannot mix custom and auto-generated MSAs in the same input!"
        raise ValueError(msg)

    # If no chains parsed fail
    if not chains:
        msg = "No chains parsed!"
        raise ValueError(msg)

    # Create tables
    atom_data = []
    bond_data = []
    # in our ab design, the bond_data is always empty due to in the parse_polymer function, we don't consider bonds, therefore, we can skip this
    
    res_data = []
    # res_data is a list of tuples, each tuple contains the information of a residue; it includes all residues in all chains
    chain_data = []
    # chain_data is a list of tuples, each tuple contains the information of a chain; it includes all chains
    
    
    # Convert parsed chains to tables
    atom_idx = 0
    res_idx = 0
    asym_id = 0
    sym_count = {}
    chain_to_idx = {}
    # chain_to_idx is a dictionary that maps chain name to its asym_id (index of the chain among all chains)

    # Keep a mapping of (chain_name, residue_idx, atom_name) to atom_idx
    atom_idx_map = {}
    h_chain_id = l_chain_id = None
    name_str = name.split("_")
    if len(name_str) == 4:
        h_chain_name = name_str[1]
        l_chain_name = name_str[2]
    else:
        h_chain_name = l_chain_name = None
    
    for asym_id, (chain_name, chain) in enumerate(chains.items()):
        if chain_name == h_chain_name:
            h_chain_id = asym_id
        if chain_name == l_chain_name:
            l_chain_id = asym_id

        # Compute number of atoms and residues
        res_num = len(chain.residues)
        # res_num is the number of residues in the chain
        atom_num = sum(len(res.atoms) for res in chain.residues)
        # atom_num is the total number of atoms in the chain

        # Find all copies of this chain in the assembly
        entity_id = int(chain.entity)
        sym_id = sym_count.get(entity_id, 0)
        chain_data.append(
            (
                chain_name,
                # chain_name is the name of the chain
                chain.type,
                # chain.type is the type of the chain. In our ab design, it is always 0
                entity_id,
                # entity_id is the index of the entity among all entities
                sym_id,
                # sym_id is the index of the chain among all chains with the same entity
                asym_id,
                # asym_id is the index of the chain among all chains
                atom_idx,
                atom_num,
                res_idx,
                res_num,
            )
        )
        chain_to_idx[chain_name] = asym_id
        # chain_to_idx is a dictionary that maps chain name to its asym_id (index of the chain among all chains)
        sym_count[entity_id] = sym_id + 1
        # sym_count is a dictionary that counts the number of chains with the same entity
        # Add residue, atom, bond, data
        for res in chain.residues:
            atom_center = atom_idx + res.atom_center
            # atom_center is the index of the center atom of the current residue
            atom_disto = atom_idx + res.atom_disto
            # atom_disto is the index of the distal atom of the current residue
            res_data.append(
                (
                    res.name,
                    # res.name is the name of the residue
                    res.type,
                    # token id of the residue
                    res.idx,
                    # index of the residue in the chain
                    atom_idx,
                    # index of the first atom of the residue
                    len(res.atoms),
                    # number of atoms included in the residue
                    atom_center,
                    # index of the center atom of the residue
                    atom_disto,
                    # index of the distal atom of the residue
                    res.is_standard,
                    # whether the residue is standard, in our ab design, it is always True
                    res.is_present,
                    # whether the residue is present, in our ab design, it is always True
                    res.is_spec_residue,
                )
            )

            # in our ab design, res.bonds is always empty due to in the parse_polymer function, we don't consider bonds, therefore, we can skip this
            for bond in res.bonds:
                atom_1 = atom_idx + bond.atom_1
                atom_2 = atom_idx + bond.atom_2
                bond_data.append((atom_1, atom_2, bond.type))

            # traverse all atoms in the residue
            for atom in res.atoms:
                # Add atom to map
                atom_idx_map[(chain_name, res.idx, atom.name)] = (
                    asym_id,
                    res_idx,
                    atom_idx,
                )
                # atom_idx_map is a dictionary that maps (chain_name, residue_idx, atom_name) to (asym_id, res_idx, atom_idx)
                """
                asym_id: index of the chain among all chains
                res_idx: index of the residue among all residues in all chains
                atom_idx: index of the atom among all atoms in all chains
                """
                
                # Add atom to data
                atom_data.append(
                    (
                        convert_atom_name(atom.name),
                        # convert the original atom name to a tuple of 4 integers, e.g. CD1 -> (35, 36, 17, 0)
                        atom.element,
                        # element is the atomic number of the atom, 也就是元素周期表中的序号
                        atom.charge,    
                        # charge is the formal charge of the atom
                        atom.coords,
                        # always (0, 0, 0) after the parse_polymer function
                        atom.conformer,
                        # a reference coordinates of the atom
                        atom.is_present,
                        # is always True after the parse_polymer function
                        atom.chirality,
                        atom.is_spec_atom,
                    )
                )
                atom_idx += 1

            res_idx += 1

    # in our ab design, we don't consider constraints, therefore, we can skip this
    # Parse constraints
    connections = []
    pocket_binders = []
    pocket_residues = []
    constraints = schema.get("constraints", [])
    for constraint in constraints:
        if "bond" in constraint:
            if "atom1" not in constraint["bond"] or "atom2" not in constraint["bond"]:
                msg = f"Bond constraint was not properly specified"
                raise ValueError(msg)

            c1, r1, a1 = tuple(constraint["bond"]["atom1"])
            c2, r2, a2 = tuple(constraint["bond"]["atom2"])
            c1, r1, a1 = atom_idx_map[(c1, r1 - 1, a1)]  # 1-indexed
            c2, r2, a2 = atom_idx_map[(c2, r2 - 1, a2)]  # 1-indexed
            connections.append((c1, c2, r1, r2, a1, a2))
        elif "pocket" in constraint:
            if "binder" not in constraint["pocket"] or "contacts" not in constraint["pocket"]:
                msg = f"Pocket constraint was not properly specified"
                raise ValueError(msg)

            binder = constraint["pocket"]["binder"]
            contacts = constraint["pocket"]["contacts"]

            if len(pocket_binders) > 0:
                if pocket_binders[-1] != chain_to_idx[binder]:
                    msg = f"Only one pocket binders is supported!"
                    raise ValueError(msg)
                else:
                    pocket_residues[-1].extend([
                        (chain_to_idx[chain_name], residue_index - 1) for chain_name, residue_index in contacts
                    ])

            else:
                pocket_binders.append(chain_to_idx[binder])
                pocket_residues.extend(
                    [(chain_to_idx[chain_name],residue_index-1) for chain_name,residue_index in contacts]
                )
        else:
            msg = f"Invalid constraint: {constraint}"
            raise ValueError(msg)
    # Since we don't consider constraints, after the above code, connections, pocket_binders, pocket_residues are all empty

    # Convert into datatypes
    atoms = np.array(atom_data, dtype=Atom)
    # all atoms in all chains
    bonds = np.array(bond_data, dtype=Bond)
    # in our ab design, the bond_data is always empty due to in the parse_polymer function, we don't consider bonds, therefore, we can skip this
    residues = np.array(res_data, dtype=Residue)
    # all residues in all chains
    chains = np.array(chain_data, dtype=Chain)
    # all chains
    interfaces = np.array([], dtype=Interface)
    # in our ab design, we don't consider interfaces, therefore, we can skip this
    connections = np.array(connections, dtype=Connection)
    # in our ab design, we don't consider constraints, therefore, we can skip this and connections is always empty
    mask = np.ones(len(chain_data), dtype=bool)
    # mask is a boolean array of the same length as chain_data

    data = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        connections=connections,
        interfaces=interfaces,
        mask=mask,
    )

    # Create metadata
    struct_info = AntibodyInfo(
        num_chains=len(chains),
        H_chain_id=h_chain_id,
        L_chain_id=l_chain_id
    )
        
    chain_infos = []
    for chain in chains:
        chain_info = ChainInfo(
            chain_id=int(chain["asym_id"]),
            chain_name=chain["name"],
            mol_type=int(chain["mol_type"]),
            cluster_id=-1,
            msa_id=chain_to_msa[chain["name"]],
            num_residues=int(chain["res_num"]),
            valid=True,
            entity_id=int(chain["entity_id"]),
        )
        chain_infos.append(chain_info)
        if chain["name"] == h_chain_name or chain["name"] == l_chain_name:
            seq_gt += entity_to_gt[chain["entity_id"]]
            spec_masks += entity_to_spec_mask[chain["entity_id"]]

    if len(seq_gt) == 0: 
        seq_gt = None
    if len(spec_masks) == 0:
        spec_masks = None

    # in our ab design, we don't consider pocket binders and pocket residues, therefore, we can skip this
    options = InferenceOptions(
        binders=pocket_binders,
        pocket=pocket_residues
    )

    record = Record(
        id=name,
        # name is the name of the yaml file
        structure=struct_info,
        chains=chain_infos,
        interfaces=[],
        inference_options=options,
    )
    return Target(
        record=record,
        structure=data,
        sequences=entity_to_seq,
    ), dict(
        seq_gt=seq_gt,
        spec_mask=spec_masks,
        entity_to_gt=entity_to_gt,
    )
