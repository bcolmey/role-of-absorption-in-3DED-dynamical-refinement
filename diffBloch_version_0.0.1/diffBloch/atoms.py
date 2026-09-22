import torch
import torch.nn as nn

from ase import Atoms as ASEAtoms
from ase.io.cif import parse_cif
from ase.geometry.analysis import Analysis

from diffpy.structure.spacegroups import GetSpaceGroup
from diffpy.structure.symmetryutilities import SymmetryConstraints

import numpy as np
import pandas as pd

import gemmi

import re
import warnings

from CifFile import ReadCif

from diffBloch.utils import element_from_atomic_number

class Uij_layer(nn.Module):
    def __init__(self, thermal_displacements, A, A_inv, D_star, U_iso_flag, constraints=None):
        """
        Initializes the Uij layer. Assumes that Uij is in Ucif format.
        Returns Uij in Ustar format
        
        Params:
        - thermal_displacements: initial thermal displacements, can be either Uiso or Uij
        - A: orthogonalization matrix
        - D_star: D_star matrix (diagonal entries are a*, b*, c*)
        - U_iso_flag: flag to convert Uiso to Uij
        - constraints: list of lists of tuples of constraints to apply to the Uij matrix in format (i,j,k,l, value) 
            where U[i,j] = value*U[k,l]
        """
        self.A = A
        self.A_inv = A_inv
        self.D_star = D_star
        super(Uij_layer, self).__init__()
        #seperate config for Uij layer?
        thermal_disp_parameters = []
        for U in thermal_displacements:
            if U.dim() == 0:
                thermal_disp_parameters.append(U)
            elif U.dim() == 2 and U_iso_flag:
                U_cart = self.A @ self.D_star @ U @ (self.A @ self.D_star).T
                U_iso = self.calculate_U_iso_from_U_aniso(U_cart)
                warnings.warn("Converting U_ani to U_iso. Change config cfg.isotropic_displacements_only to false if anistropic displacements required.", UserWarning)
                thermal_disp_parameters.append(U_iso)
            else:
                # store Uij_cif in cholesky form
                U_cholesky = torch.linalg.cholesky(U)
                thermal_disp_parameters.append(U_cholesky)

        # thermal disps to optimise, might be a combo of Uiso and Uij, can expand in forward
        self.thermal_displacements = nn.ParameterList(thermal_disp_parameters)

        # constraints to apply to Uij
        if constraints:
            self.constraints = constraints
        else:
            # empty list of lists of length of thermal_displacements
            self.constraints = [[] for _ in range(len(thermal_displacements))]

    def forward(self):
        """
        Forward pass of the Uij layer.
        Apply constraints to Uij if needed, expand Uiso to Uij form.
        Return Uij matrices in Cartesian form
        """
        #way to vectorise?
        #need to be careful in expansion because units handled differently
        #Ustar thermal displacements that will be returned to atoms
        U_star = []
        for i, U in enumerate(self.thermal_displacements):
            #get Uij equivalent form of Uiso
            if U.dim() == 0:
                Uiso_cart_Uij_form = self.expand_Uij(U)
                #store as U_star
                Uij_star = self.A_inv @ Uiso_cart_Uij_form @ self.A_inv.T 
                U_star.append(Uij_star)
            elif U.dim() == 2:
                #expand from cholesky form
                Uij_cif = self.expand_Uij(U)
                # apply constraints
                Uij_cif_constrained = self.apply_constraints(Uij_cif, self.constraints[i])
                #convert from Ucif to Ucart
                Uij_cart = self.A @ self.D_star @ Uij_cif_constrained @ (self.A @ self.D_star).T
                #convert from Ucif to Ustar
                Uij_star = self.A_inv @ Uij_cart @ self.A_inv.T
                #store as U_star
                U_star.append(Uij_star)
            else:
                raise ValueError(f"Invalid thermal displacement parameter {U}")
        
        return torch.stack(U_star, dim=0)
    
    def apply_constraints(self, Uij, constraints):
        # TODO check for memory leaks?
        # TODO vectorise
        # Apply each constraint
        for i, j, k, l, coeff in constraints:
            Uij[i, j] = coeff * Uij[k, l]
        return Uij

    def calculate_U_iso_from_U_aniso(self, Uij):
        """
        Calculate the isotropic Debye-Waller factor from anisotropic displacement parameters (ADPs).

        Parameters:
        - Uij (ndarray): 3x3 matrix of atomic displacement parameters (Uij) in cartesian_coordinates.
        - unit_cell (ndarray): 3x3 matrix of lattice vectors. If None, the unit cell of the Atoms object is used.
        """
    
        assert Uij.shape == (3, 3), "Uij must be a 3x3 matrix"

        U_iso =  torch.trace(Uij) / 3.0

        return U_iso
    
    def expand_Uij(self, thermal_displacement):
        """
        Expand the Uij matrix from either cholesky form or Uiso.
        
        Parameters:
        thermal_displacement
        
        Returns:
        torch.tensor: Uij_cart matrix

        equation 9 from https://journals.iucr.org/j/issues/2002/04/00/ks0128/index.html

        """
        if thermal_displacement.dim() == 0:
            # assume isotropic form
            return thermal_displacement * torch.eye(3, dtype=thermal_displacement.dtype, device=thermal_displacement.device)
        elif thermal_displacement.dim() == 2:
            # assume cholesky form
            return thermal_displacement @ thermal_displacement.T
        else:
            raise ValueError(f"Invalid thermal displacement parameter {thermal_displacement}")

    def get_all_Uij_tensor(self):
        """
        Get all Uij tensors as a tensor.
        """
        return torch.stack([self.expand_Uij(x) for x in self.thermal_displacements], dim=0)
    
    def __repr__(self):
        return f"Uij_layer(thermal_displacements={self.thermal_displacements}, constraints={self.constraints})"


class Atoms(nn.Module):
    def __init__(self, cfg):
        """
        Initializes the Atoms class.
        
        Params:
        - structure_path: path to .cif file (optional)
        - device: 'cpu' or 'gpu' to specify the computation device
        """
        super(Atoms, self).__init__()
        
        # Device setting
        self.device = cfg.experiment.device
        self.isotropic = cfg.experiment.isotropic_displacements_only
        self.cfg = cfg
        if not cfg.data.cif_file_path:
            raise ValueError("The CIF file path is null or empty.")
        asu_positions, bond_pairs, spacegroup, centering, unit_cell, abg_angles, asu_numbers, asu_atom_labels, thermal_displacements, thermal_displacement_type = self.load_structure_asymmetric_unit(cfg.data.cif_file_path)
        
        self.asu_positions, self.bond_pairs, self.spacegroup, self.centering, self.unit_cell, self.abg_angles, self.asu_numbers, self.asu_atom_labels, self.thermal_displacement_type = asu_positions, bond_pairs, spacegroup, centering, unit_cell, abg_angles, asu_numbers, asu_atom_labels, thermal_displacement_type

        # init bond/angle constraints
        if self.cfg.data.restraints_file_path:
            self.bond_constraints_idxs, self.bond_constraints, self.angle_constraints_idxs, self.angle_constraints, self.plane_constraints_idxs = self.load_constraints_from_mogul_search_results(self.cfg.data.restraints_file_path)
        else:
            # TODO might need to change to list
            self.bond_constraints_idxs, self.bond_constraints, self.angle_constraints_idxs, self.angle_constraints, self.plane_constraints_idxs = None, None, None, None, []

        # init Uij_layer here
        A = torch.tensor(self.orthogonalization_matrix(), dtype=torch.float64, device=self.device)
        A_inv = torch.tensor(self.inverse_orthogonalization_matrix(), dtype=torch.float64, device=self.device)
        D_star = torch.tensor(self.D_star_matrix(), dtype=torch.float64, device=self.device)
        # use diffpy to get constraints
        Uij_constraints, position_constraints = self.get_constraints_from_diffpy(asu_positions.detach().cpu().numpy())
        
        self.asu_positions = nn.Parameter(self.asu_positions, requires_grad=True)
        self.Uij_layer = Uij_layer(thermal_displacements, A, A_inv, D_star, cfg.experiment.isotropic_displacements_only, Uij_constraints)
        self.asu_positions_sym_mask = torch.tensor(position_constraints, dtype=torch.float64, device=self.device)

        # stop grads if needed
        if not cfg.experiment.optimize_asu_positions:
            self.asu_positions.requires_grad = False
        if not cfg.experiment.optimize_thermal_displacements:
            for param in self.Uij_layer.parameters():
                param.requires_grad = False

        #if uani, symmetry constrain
        print(f"Loaded structure from {self.cfg.data.cif_file_path}")
        print(f"Asymmetric unit positions: {self.asu_positions}")
        print(f"Atomic numbers: {self.asu_numbers}")
        print(f"Atomic labels: {self.asu_atom_labels}")
        print(f"ASU Symmetry constraints,(1 if allowed to move): {self.asu_positions_sym_mask}")
        print(f"Bond constraints: {self.bond_constraints_idxs}, {self.bond_constraints}")
        print(f"Angle constraints: {self.angle_constraints_idxs}, {self.angle_constraints}")
        print(f"Plane constraints: {self.plane_constraints_idxs}")
        print(f"Spacegroup: {self.spacegroup}")
        print(f"Unit cell lattice vectors a,b,c: {self.unit_cell}")
        print(f"lattice vector lengths: {self.cell_lengths()}")
        print(f"Angles (alpha, beta, gamma) between lattice vectors in degrees: {self.abg_angles}")
        print(f"Thermal displacements: {self.Uij_layer}")

    def load_structure_asymmetric_unit(self, structure_path):
        """
        Load atomic structure from a .cif file and store the atomic data in the class.
        """
        thermal_displacements = []
        thermal_displacement_type = []
        asu_atom_labels = []
        for block in parse_cif(structure_path):
            unsymmetrized_atoms = block.get_unsymmetrized_structure()   # ASE atoms object
            # remove hydrogens
            if not self.cfg.data.load_hydrogens:
                structure_without_hydrogens = [atom for atom in unsymmetrized_atoms if atom.number != 1]
                unsymmetrized_atoms = ASEAtoms(structure_without_hydrogens, cell=unsymmetrized_atoms.cell, pbc=unsymmetrized_atoms.pbc)

            unit_cell = np.array(unsymmetrized_atoms.cell.array, dtype=np.float64)
           
            abg_angles = np.array(unsymmetrized_atoms.cell.angles(), dtype=np.float64)
            if '_symmetry_space_group_name_h-m' not in block:
                raise ValueError("No Hermann-Mauguin (HM) notation found in the CIF file. "
                                "Our code requires a HM spacegroup to be passed. "
                                "Please check the CIF file and try again.")
            spacegroup_hm = block['_symmetry_space_group_name_h-m']
            if spacegroup_hm == 'P 21/n':
                print(f'completed:{spacegroup_hm}')
                spacegroup_hm = 'P1121/n'
            spacegroup = GetSpaceGroup(spacegroup_hm)
            #TODO Tidy this up so that all spacegroup handled by diffpy
            spacegroup_ase = block.get_spacegroup(subtrans_included=False)
            centering = self.get_centering_from_symbol(spacegroup_ase.symbol)
            ana = Analysis(unsymmetrized_atoms) # ASE analysis object
            bond_list = ana.unique_bonds[0] # get unique bonds from ASE analysis object (using cartesian coordinates)
            
            # Iterate over atoms in the structure for thermal displacements
            includes_thermal_displacements = True
            if "_atom_site_thermal_displace_type" not in block:
                print("No thermal displacement parameters found in the CIF file, defaulting to Uiso=0 for all.")
                includes_thermal_displacements = False
            k = 0   # counter for anisotropic displacement parameters
            for i, atom_site_label in enumerate(block["_atom_site_label"]):
                # don't load hydrogens if specified
                if not self.cfg.data.load_hydrogens and block["_atom_site_type_symbol"][i] == "H":
                    continue
                asu_atom_labels.append(atom_site_label)
                if includes_thermal_displacements:
                    if block["_atom_site_thermal_displace_type"][i] == "Uiso":
                        thermal_displacement = torch.tensor(float(block["_atom_site_u_iso_or_equiv"][i]), dtype=torch.float64, device=self.device)
                    elif block["_atom_site_thermal_displace_type"][i] == "Uani":
                        # Construct Uij matrix for anisotropic displacement parameters
                        Uij = self.construct_Uij_matrix(block, k)
                        k+=1
                        thermal_displacement = torch.tensor(Uij, dtype=torch.float64, device=self.device)
                    # Append the displacement to the list, assumed thermal disps are in Uciff format
                    thermal_displacement_type.append(block["_atom_site_thermal_displace_type"][i])
                    thermal_displacements.append(thermal_displacement)
                else:
                    thermal_displacements.append(torch.tensor(0.0, dtype=torch.float64, device=self.device))
                    thermal_displacement_type.append("Uiso")
        # These are scaled positions
        asu_positions = torch.tensor(unsymmetrized_atoms.get_scaled_positions(), dtype=torch.float64, device=self.device)
        bond_pairs = torch.tensor(self.convert_bond_list_to_bond_pairs(bond_list), dtype=torch.int, device=self.device)
        asu_numbers = torch.tensor(unsymmetrized_atoms.numbers, dtype=torch.int, device=self.device)
        
        return asu_positions, bond_pairs, spacegroup, centering, unit_cell, abg_angles, asu_numbers, asu_atom_labels, thermal_displacements, thermal_displacement_type
    
    def load_constraints_from_mogul_search_results(self, constraints_file):
        """
        Load atom constraints from a file. 
        Reads the bond and angle constraints from a Mogul search results file and returns:
        Returns:
        - bond_constraints_idxs: tensor of bond indices (n_bonds, 2) 
        - bond_constraints: tensor of bond constraints, mean and std. dev. (n_bonds, 2)
        - angle_constraints_idxs: tensor of angle indices (n_angles, 3)
        - angle_constraints: tensor of angle constraints, mean and std. dev. (n_angles, 2)
        - plane_constraints_idxs: list of lists of atom indices for plane constraints
        """
        constraints_df = pd.read_csv(constraints_file)
        # bonds
        bond_constraints = constraints_df[constraints_df['Type'] == 'bond']
        # expand out the atoms in the bond constraints using spaces in the Fragment column
        bond_constraints[['atom_1', 'atom_2']] = pd.DataFrame(bond_constraints['Fragment'].apply(lambda x: x.split(' ')).to_list(), index=bond_constraints.index)
        # get corresponding atom indices
        bond_constraints['atom_1_idx'] = bond_constraints['atom_1'].apply(lambda x: self.asu_atom_labels.index(x))
        bond_constraints['atom_2_idx'] = bond_constraints['atom_2'].apply(lambda x: self.asu_atom_labels.index(x))
        # save as idx tensor
        bond_constraints_idxs = torch.tensor(bond_constraints[['atom_1_idx', 'atom_2_idx']].values, dtype=torch.long, device=self.device)
        # save mean and std as another tensor
        bond_constraints = torch.tensor(bond_constraints[['Mean', 'Std. dev.']].values, dtype=torch.float, device=self.device)

        # angles
        angle_constraints = constraints_df[constraints_df['Type'] == 'angle']
        angle_constraints[['atom_1', 'atom_2', 'atom_3']] = pd.DataFrame(angle_constraints['Fragment'].apply(lambda x: x.split(' ')).to_list(), index=angle_constraints.index)
        angle_constraints['atom_1_idx'] = angle_constraints['atom_1'].apply(lambda x: self.asu_atom_labels.index(x))
        angle_constraints['atom_2_idx'] = angle_constraints['atom_2'].apply(lambda x: self.asu_atom_labels.index(x))
        angle_constraints['atom_3_idx'] = angle_constraints['atom_3'].apply(lambda x: self.asu_atom_labels.index(x))
        angle_constraints_idxs = torch.tensor(angle_constraints[['atom_1_idx', 'atom_2_idx', 'atom_3_idx']].values, dtype=torch.long, device=self.device)
        angle_constraints = torch.tensor(angle_constraints[['Mean', 'Std. dev.']].values, dtype=torch.float, device=self.device)

        # planes
        plane_constraints = constraints_df[constraints_df['Type'] == 'ring']
        plane_constraints['atoms'] = plane_constraints['Fragment'].apply(lambda x: x.split(' '))
        plane_constraints['atom_idxs'] = plane_constraints['atoms'].apply(lambda x: [self.asu_atom_labels.index(atom) for atom in x])
        plane_constraints_idxs = plane_constraints['atom_idxs'].values.tolist()

        return bond_constraints_idxs, bond_constraints, angle_constraints_idxs, angle_constraints, plane_constraints_idxs

    def get_constraints_from_diffpy(self, asu_positions):
        """
        Get constraints on the position and thermal displacement parameters from diffpy.
        TODO this could be parallelised
        Parameters:
        - asu_positions: Scaled atomic positions in the asymmetric unit.
        Returns:
        - Uij_constraints: List of constraints on the thermal displacement parameters.
        - position_constraints: List of constraints on the atomic positions. Boolean mask, 1 if allowed to move, 0 if not.
        """
        # Mapping from Uij string keys to matrix indices
        uij_index_map = {
            'U11': (0, 0),
            'U22': (1, 1),
            'U33': (2, 2),
            'U12': (0, 1),
            'U13': (0, 2),
            'U23': (1, 2),
        }

        def _parse_constraint(formula):
            """
            Parse the constraint formula into indices and a scaling factor.
            :param formula: String formula from the API (e.g., '0.5*U110', 'U130')
            :return: Tuple (i, j, k, l, coeff), representing Uij[i, j] = coeff * Uij[k, l]
            """
            # Regex to match optional coefficient and Uij element
            match = re.match(r'(?:(\d*\.\d+|\d+)\*)?(U\d\d)', formula)
            if not match:
                raise ValueError(f"Invalid formula format: {formula}")
            
            coeff_str, uij_str = match.groups()
            coeff = float(coeff_str) if coeff_str else 1.0  # Default to 1 if no coefficient is provided
            k, l = uij_index_map[uij_str[:3]]  # Map Uij to its indices
            
            return k, l, coeff

        def _convert_diffpy_uij_constraints(api_constraints):
            """
            Convert the list of anisotropic atomic displacement formula dictionaries into constraints
            :param api_constraints: List of dictionaries from API containing formulas
            :return: List of constraints as (i, j, k, l, coeff)
            """
            constraints = []
            for uij_key, formula in api_constraints.items():
                if formula != '0':  # Ignore '0' constraints (implies no relationship)
                    i, j = uij_index_map[uij_key]  # Get the Uij indices for the key (U11, U12, etc.)
                    k, l, coeff = _parse_constraint(formula)  # Parse the formula (coeff, Uij)
                    if i == k and j == l and coeff == 1.0:  # Skip constraints that are already identity
                        continue
                    constraints.append((i, j, k, l, coeff))  # Append the constraint

            return constraints

        def _convert_diffpy_pos_constraints(api_constraints):
            """
            Converts dictionary of atom position symmetry constraints to constraint list
            In DiffPy, a position fixed if there is only a scalar value in the formula (no x,y,z)
            We convert this to a boolean mask, 1 if allowed to move, 0 if not. 
            e.g. {'x': 'x0', 'y': '+0.5', 'z': 'z0'} -> [1, 0, 1]
            :param api_constraints: Dictionary of atom position symmetry constraints
            """
            return [1 if any([c in 'xyz' for c in x]) else 0 for x in api_constraints.values()]

        # Get symmetry constraints from diffpy
        symcon = SymmetryConstraints(self.spacegroup, positions=asu_positions)
        Uij_constraints = [_convert_diffpy_uij_constraints(x) for x in symcon.Ueqns]
        position_constraints = [_convert_diffpy_pos_constraints(x) for x in symcon.poseqns]

        return Uij_constraints, position_constraints

    def convert_bond_list_to_bond_pairs(self, bond_list):
        """
        Convert a list of bonded atoms to a list of bond pairs.
        e.g. [[1], [3,4], [], [], []] -> [[0,1], [1,3], [1,4]]
        """
        bond_pairs = []
        for i, bonded_atoms in enumerate(bond_list):
            for j in bonded_atoms:
                bond_pairs.append([i, j])
        return bond_pairs

    def forward(self, symprec=1e-3, onduplicates="error"):
        """
        Forward process performs the asymmetric unit expansion.
        
        Params:
        - symprec: Tolerance for detecting equivalent sites.
        - onduplicates: Action to take if duplicates are found ('keep', 'replace', 'warn', 'error').

        Returns:
        - expanded_positions: Expanded scaled atomic positions as a PyTorch tensor.
        - expanded_atomic_numbers: Expanded atomic numbers as a NumPy array.
        - expanded_disps: Expanded thermal displacements as a PyTorch tensor.
        - occupancy: Occupancy of each atom as a NumPy array.
        """

        # Perform symmetry expansion
        expanded_positions, atom_id, expanded_disps, expanded_atomic_numbers = self.asu_expansion(symprec, onduplicates)
        # TODO, currently only works as numpy but will eventually make differentiable
        # self.occupancy = torch.ones(len(self.positions), device=self.positions.device)
        # self.occupancy = self.occupancy.detach().cpu().numpy()
        occupancy = np.ones(len(expanded_positions), dtype=np.float64)
        
        warnings.warn("Default behavior currently assumes occupancy of 1 for every atom.")

        return expanded_positions, expanded_atomic_numbers, expanded_disps, occupancy

    def asu_expansion(self, symprec=1e-3, onduplicates="error"):
        """
        Expand the asymmetric unit to the full unit cell using symmetry operations.
        """
        device = self.asu_positions.device
        print(f'device:{device}')
        kinds = []
        sites = []
        sites2 = []
        expanded_atomic_numbers = []
        expanded_thermal_disps = []
        Uijs = self.Uij_layer() # get Uij matrices in Ustar format
        #TODO below hack to get correct Uijs for saving, should improve integration
        self.Uij_star = Uijs.detach()

        # zero out grad using mask using torch stop grad addition trick
        positions = self.asu_positions * self.asu_positions_sym_mask + self.asu_positions.detach() * (1 - self.asu_positions_sym_mask)

        # Iterate through scaled asu positions and apply symmetry operations
        for kind, pos in enumerate(positions):
            
            # Uij should be in Ustar format here
            Uij = Uijs[kind]
            for symop in self.spacegroup.iter_symops():
                rot = symop.R
                trans = symop.t
                site = torch.remainder(
                    torch.matmul(torch.tensor(rot, dtype=torch.float64, device=device), pos)
                    + torch.tensor(trans, dtype=torch.float64, device=device),
                    1.0
                )
                rot = torch.tensor(rot, dtype=torch.float64, device=device)
                A = torch.tensor(self.orthogonalization_matrix(), dtype=torch.float64, device=device)
                rotated_Uij_star = rot @ Uij @ rot.T # rotate Uij matrix
                #TODO, should probably store as Uijstar and convert when doing bond restraint
                rotated_Uij_cart = A @ rotated_Uij_star @ A.T
                site_np = site.detach().cpu().numpy()  # Convert to NumPy for duplicate detection
                # initialise the first site
                if not sites:
                    sites.append(site)
                    sites2.append(site_np)
                    kinds.append(kind)
                    expanded_atomic_numbers.append(self.asu_numbers[kind])  # Track atomic number
                    expanded_thermal_disps.append(rotated_Uij_cart)
                    continue
                
                # Detect duplicates based on symprec 
                diff = site_np - np.array(sites2)
                mask = np.all((np.abs(diff) < symprec) | (np.abs(np.abs(diff) - 1.0) < symprec), axis=1)
                if np.any(mask):
                    inds = np.argwhere(mask).flatten()
                    for ind in inds:
                        if kinds[ind] == kind:
                            pass
                        elif onduplicates == "keep":
                            pass
                        elif onduplicates == "replace":
                            kinds[ind] = kind
                            expanded_atomic_numbers[ind] = self.asu_numbers[kind]  # Replace atomic number
                        elif onduplicates == "warn":
                            warnings.warn(
                                f"scaled_positions {kinds[ind]} and {kind} are equivalent"
                            )
                        elif onduplicates == "error":
                            raise ValueError(
                                f"scaled_positions {kinds[ind]} and {kind} are equivalent"
                            )
                        else:
                            raise ValueError('Invalid value for "onduplicates".')
                else:
                    # If no duplicates found, append the site
                    sites2.append(site_np)
                    sites.append(site)
                    kinds.append(kind)
                    expanded_atomic_numbers.append(self.asu_numbers[kind])  # Append atomic number
                    expanded_thermal_disps.append(rotated_Uij_cart)
        
        # Convert the sites list back to a tensor
        sites_tensor = torch.stack(sites, dim=0)

        # Convert expanded atomic numbers to a tensor
        expanded_atomic_numbers = torch.tensor(expanded_atomic_numbers, dtype=torch.int, device=self.device)
        
        expanded_thermal_disps = torch.stack(expanded_thermal_disps, dim=0)

        return sites_tensor, kinds, expanded_thermal_disps, expanded_atomic_numbers

    def reciprocal_cell(self, unit_cell=None):
        """
        Computes the reciprocal lattice vectors
        mimicking ASE's Cell.reciprocal() function.
        Parameters:
        - unit_cell (np.ndarray): 3x3 matrix of lattice vectors. If None, the unit cell of the Atoms object is used.
        
        Returns:
        - A 3x3 tensor containing the reciprocal lattice vectors.
        """
        unit_cell = self.unit_cell if unit_cell is None else unit_cell
        # Ensure the unit_cell is a 3x3 tensor
        assert unit_cell.shape == (3, 3), "Unit cell must be a 3x3 matrix"
        
        # Compute the pseudoinverse of the unit cell
        pinv_unit_cell = np.linalg.pinv(unit_cell).transpose()  # Take pseudoinverse and then transpose
        
        return pinv_unit_cell
    
    def D_star_matrix(self):
        d_star = np.zeros((3,3))
        d_star[0,0] = np.linalg.norm(self.reciprocal_cell()[0])
        d_star[1,1] = np.linalg.norm(self.reciprocal_cell()[1])
        d_star[2,2] = np.linalg.norm(self.reciprocal_cell()[2])
        
        return d_star

    def A_matrix(self):
        """ 
        Compute the A matrix as defined in equation 50 of Trueblood et al. Acta Cryst. (1996). A52, 770-781.
        The matrix A is used for transforming between fractional and Cartesian coordinates in a specific 
        crystallographic context.
        
        A = [[a, b * cos(gamma), c * cos(beta)],
            [0, b * sin(gamma), -c * sin(beta) * cos(alpha_star)],
            [0, 0, 1 / c_star]]
        
        where:
        a, b, c are the lattice constants,
        alpha, beta, gamma are the lattice angles, 
        alpha_star and c_star are reciprocal angles/lengths.
        """
        
        # Get the lattice constants and angles
        a, b, c = self.cell_lengths()  # a, b, c lengths of lattice vectors
        alpha, beta, gamma = np.deg2rad(self.abg_angles)  # Convert degrees to radians
        V = self.cell_volume()

        # Compute the reciprocal lattice constant c_star and angle alpha_star
        c_star = np.linalg.norm(np.cross(self.unit_cell[2], self.unit_cell[1]) / V)
        
        cos_alpha_star = (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / (np.sin(beta) * np.sin(gamma))
        alpha_star = np.arccos(cos_alpha_star)

        # Create the A matrix
        A = np.zeros((3, 3))
        A[0, 0] = a
        A[0, 1] = b * np.cos(gamma)
        A[0, 2] = c * np.cos(beta)
        A[1, 1] = b * np.sin(gamma)
        A[1, 2] = -c * np.sin(beta) * np.cos(alpha_star)
        A[2, 2] = 1 / c_star

        return A

    def metric_tensor(self):
        """
        Compute the metric tensor for the unit cell.
        """
        # Get the lattice vectors and angles
        a, b, c = self.unit_cell

        # Compute the metric tensor
        G = np.zeros((3, 3))
        G[0,0] = np.dot(a,a)
        G[1,1] = np.dot(b,b)
        G[2,2] = np.dot(c,c)
        G[0,1] = np.dot(b,a)
        G[0,2] = np.dot(c,a)
        G[1,0] = np.dot(a,b)
        G[1,2] = np.dot(c,b)
        G[2,0] = np.dot(a,c)
        G[2,1] = np.dot(b,c)
        
        return G

    def orthogonalization_matrix(self):
        """
        Compute the orthogonalization matrix for converting fractional coordinates to Cartesian coordinates.
        Uses tolerance to zero out near-zero trigonometric calculations.
        """
        # Get the lattice vectors and angles
        a, b, c = np.array(self.cell_lengths(), dtype=np.float64)
        alpha, beta, gamma = np.deg2rad(np.array(self.abg_angles, dtype=np.float64))
        
        # Compute the volume of the cell
        V = np.float64(self.cell_volume())
        
        # Compute the orthogonalization matrix
        O = np.zeros((3, 3), dtype=np.float64)
        
        # Tolerance for considering a value effectively zero
        tol = 1e-14
        
        O[0, 0] = a
        
        # Zero out values that are very close to zero
        cos_gamma = np.cos(gamma)
        cos_gamma = 0.0 if np.abs(cos_gamma) < tol else cos_gamma
        O[0, 1] = b * cos_gamma
        
        cos_beta = np.cos(beta)
        cos_beta = 0.0 if np.abs(cos_beta) < tol else cos_beta
        O[0, 2] = c * cos_beta
        
        sin_gamma = np.sin(gamma)
        sin_gamma = 0.0 if np.abs(sin_gamma) < tol else sin_gamma
        O[1, 1] = b * sin_gamma
        
        cos_alpha = np.cos(alpha)
        cos_beta_cos_gamma = np.cos(beta) * np.cos(gamma)
        
        cos_alpha = 0.0 if np.abs(cos_alpha) < tol else cos_alpha
        cos_beta_cos_gamma = 0.0 if np.abs(cos_beta_cos_gamma) < tol else cos_beta_cos_gamma
        
        if np.abs(sin_gamma) >= tol:
            O[1, 2] = c * (cos_alpha - cos_beta_cos_gamma) / sin_gamma
        
        O[2, 2] = V / (a * b * sin_gamma)
        
        return O

    def inverse_orthogonalization_matrix(self):
        """
        Compute the inverse of the orthogonalization matrix.
        """
        return np.linalg.inv(self.orthogonalization_matrix())

    def cell_lengths(self):
        """
        Compute the lengths of the lattice vectors.
        """
        return np.linalg.norm(self.unit_cell, axis=1)

    def cell_volume(self) -> float:
        """Get the volume of the cell, adapted from ASE.

        If there are less than 3 lattice vectors, return 0."""
        # Fail or 0 for <3D cells?
        # Definitely 0 since this is currently a property.
        # I think normally it is more convenient just to get zero
        return np.abs(np.linalg.det(self.unit_cell))

    def get_asu_cartesian_coords(self):
        """
        Get the cartesian coordinates of the asymmetric unit
        """
        return torch.matmul(torch.tensor(self.unit_cell, device=self.device).T, self.asu_positions.T).T
 
    def save_cif(self, save_path=None):
        doc = gemmi.cif.Document()
        block = doc.add_new_block('data_refined')

        # Unit cell and space group
        a, b, c = np.linalg.norm(self.unit_cell, axis=1)
        alpha, beta, gamma = self.abg_angles
        cell = [a, b, c, alpha, beta, gamma]

        block.set_pair('_cell_length_a', f'{cell[0]:.5f}')
        block.set_pair('_cell_length_b', f'{cell[1]:.5f}')
        block.set_pair('_cell_length_c', f'{cell[2]:.5f}')
        block.set_pair('_cell_angle_alpha', f'{cell[3]:.5f}')
        block.set_pair('_cell_angle_beta', f'{cell[4]:.5f}')
        block.set_pair('_cell_angle_gamma', f'{cell[5]:.5f}')
        block.set_pair('_cell_volume', f'{self.cell_volume():.5f}')

        shortname= self.spacegroup.short_name
        block.set_pair('_symmetry_space_group_name_H-M', f"'{shortname}'")
        spacegroup = gemmi.find_spacegroup_by_name(shortname)
        print(f'spacegroup:{spacegroup}')
        block.set_pair('_symmetry_Int_Tables_number', str(spacegroup.number))

        # Symmetry operations loop
        sym_ops = spacegroup.operations()
        sym_loop = block.init_loop('', ['_symmetry_equiv_pos_site_id', '_symmetry_equiv_pos_as_xyz'])

        for i, op in enumerate(sym_ops):
            sym_loop.add_row([str(i + 1), op.triplet()])


        # Atom positions loop
        atom_loop = block.init_loop('_atom_site_', [
            'label',
            'type_symbol',
            'fract_x',
            'fract_y',
            'fract_z',
            'U_iso_or_equiv',
            'thermal_displace_type'
        ])

        fract_coords = self.asu_positions.detach().cpu().numpy()
        A = torch.tensor(self.orthogonalization_matrix(), dtype=torch.float64, device=self.device)
        A_inv = torch.tensor(self.inverse_orthogonalization_matrix(), dtype=torch.float64, device=self.device)
        D_star = torch.tensor(self.D_star_matrix(), dtype=torch.float64, device=self.device)
        D_inv = torch.linalg.inv(D_star)

        for i, label in enumerate(self.asu_atom_labels):
            x, y, z = fract_coords[i]
            U_cart = A @ self.Uij_star[i] @ A.T
            U_iso = self.Uij_layer.calculate_U_iso_from_U_aniso(U_cart).item()
            atom_type = self.asu_numbers[i]
            thermal_type = self.thermal_displacement_type[i]
            atom_loop.add_row([
                label,
                element_from_atomic_number(atom_type.item()),
                f'{x:.6f}',
                f'{y:.6f}',
                f'{z:.6f}',
                f'{U_iso:.6f}',
                thermal_type
            ])

        # Anisotropic displacements loop
        if 'Uani' in self.thermal_displacement_type:
            aniso_loop = block.init_loop('_atom_site_aniso_', [
                'label',
                'U_11', 'U_22', 'U_33',
                'U_23', 'U_13', 'U_12'
            ])
            for i, label in enumerate(self.asu_atom_labels):
                if self.thermal_displacement_type[i] != 'Uani':
                    continue
                U_cart = A @ self.Uij_star[i] @ A.T
                U_cif = A_inv @ D_inv @ U_cart @ (A_inv @ D_inv).T
                aniso_loop.add_row([
                    label,
                    f'{U_cif[0, 0].item():.6f}',
                    f'{U_cif[1, 1].item():.6f}',
                    f'{U_cif[2, 2].item():.6f}',
                    f'{U_cif[1, 2].item():.6f}',
                    f'{U_cif[0, 2].item():.6f}',
                    f'{U_cif[0, 1].item():.6f}',
                ])

        # Write to file
        save_path = save_path or f'{self.cfg.data.cif_file_path}_refined.cif'
        doc.write_file(save_path)


    def get_centering_from_symbol(self, symbol):
            """
            Extract centering information from the spacegroup symbol.
            """
            if not symbol or len(symbol) == 0:
                raise ValueError("Invalid spacegroup symbol.")
            
            centering_letter = symbol[0]
            if centering_letter in ['P', 'I', 'F', 'A', 'B', 'C', 'R']:
                return centering_letter
            else:
                raise ValueError(f"Unknown centering symbol: {centering_letter}")

    def construct_Uij_matrix(self, block, atom_index):
            """
            Construct the Uij matrix for anisotropic displacement parameters (ADPs) for the atom at atom_index.
            
            Parameters:
            block (dict): Parsed CIF data block.
            atom_index (int): Index of the atom for which Uij is being constructed.
            
            Returns:
            np.ndarray: 3x3 Uij matrix.
            """
            Uij = np.zeros((3, 3))

            # Fill in the Uij matrix from the appropriate columns in the CIF block
            Uij[0, 0] = float(block["_atom_site_aniso_u_11"][atom_index])
            Uij[1, 1] = float(block["_atom_site_aniso_u_22"][atom_index])
            Uij[2, 2] = float(block["_atom_site_aniso_u_33"][atom_index])
            Uij[1, 2] = Uij[2, 1] = float(block["_atom_site_aniso_u_23"][atom_index])
            Uij[0, 2] = Uij[2, 0] = float(block["_atom_site_aniso_u_13"][atom_index])
            Uij[0, 1] = Uij[1, 0] = float(block["_atom_site_aniso_u_12"][atom_index])

            return Uij


    def __repr__(self):
        # TODO tidy
        return super().__repr__() + f"asu positions={self.asu_positions}, bond pairs={self.bond_pairs}, spacegroup={self.spacegroup}, centering={self.centering}, (unit cell) {self.unit_cell}, angles={self.abg_angles}, asu_numbers={self.asu_numbers}, asu_atom_labels={self.asu_atom_labels}"
