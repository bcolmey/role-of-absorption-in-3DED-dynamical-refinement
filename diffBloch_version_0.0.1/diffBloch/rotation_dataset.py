import pickle
import os
import ast
from typing import Sequence, Iterator

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#TODO, currently using three parsers at different points in code, reduce to just using one (gemmi?)
from pymatgen.io.cif import CifParser
from ase.io.cif import parse_cif
from gemmi import cif

def load_data(fname):
    """
    Load data from .pets file
    Returns:
    - cif_file: dictionary of data from .pets file
    """
    cif_file = CifParser(fname)
    cif_file = cif_file.as_dict()
    return cif_file

class RotationDataset(Dataset):
    """
    Dataset for experimental intensities and rotations.
    TODO 
    - mapping from rotation idx to intensities in a good format (collate function). 
    - also store and return the orientation matrices, thicknesses etc. here. 
    """

    def __init__(self, cfg, rotations: list, alphas: list, thickness: list, optim_orientations_path=None, optim_thickness_path=None):
        """
        Args:
            rotations: list of rotations
            ref_intensities_path: path to reference intensities
        """
        self.ref_intensities_path = cfg.data.pets_path
        self.dsg = cfg.data.dsg
        self.rsg = cfg.data.rsg
        # TODO use these as default with override in cfg 
        self.integration_semiangle = cfg.data.integration_semiangle
        self.rocking_curve_sampling = cfg.data.rocking_curve_sampling
        self.exp_info = get_exp_ints(self.ref_intensities_path)
        self.rotations = rotations
        self.alphas = alphas
        self.alphas_min = min(alphas) if alphas else None  
        self.alphas_max = max(alphas) if alphas else None 
        self.thicknesses = [thickness for _ in range(len(rotations))]
        self.reference_positions = self.get_reference_data()
        #scale alphas to be between -1 and 1
        self.alpha_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.alphas = self.alpha_scaler.fit_transform(np.array(self.alphas).reshape(-1, 1))
        self.rotation_axis_position, self.rc_width, self.mosaicity, self.dstarmax, self.data_collection_geometry = extract_data_params(cfg.data.pets_path)
        self.rotation_axis_correction()

        if optim_orientations_path:
            print(f'loading optim orientations from {optim_orientations_path}')
            optim_orientations_df = pd.read_csv(optim_orientations_path)
            optim_orientations_df["Orientation Matrix"] = optim_orientations_df["Orientation Matrix"].apply(lambda x: np.array(ast.literal_eval(x)))
            # overwrite rotations with optim orientations if available
            optim_rotation_idx_list = optim_orientations_df["Rotation Index"].values
            optim_orientation_list = optim_orientations_df["Orientation Matrix"].values
            for idx, rotation_idx in enumerate(optim_rotation_idx_list):
                self.rotations[rotation_idx] = optim_orientation_list[idx]
        if optim_thickness_path:
            optim_thickness_df = pd.read_csv(optim_thickness_path)
            optim_rotation_idx_list = optim_thickness_df["Rotation Index"].values
            optim_thickness_list = optim_thickness_df["Thickness"].values
            for idx, rotation_idx in enumerate(optim_rotation_idx_list):
                self.thicknesses[rotation_idx] = [optim_thickness_list[idx]]
        self.thicknesses = torch.tensor(self.thicknesses, dtype=torch.float32)
        if self.data_collection_geometry == 'continuous rotation':
            self.rocking_curve_orientations = generate_integration_rotation_matrices(self.integration_semiangle, self.rocking_curve_sampling)
        elif self.data_collection_geometry == 'precession':
            self.rocking_curve_orientations = generate_precession_rotations(self.integration_semiangle, self.rocking_curve_sampling)
        else:
            raise ValueError(
                f"Invalid data_collection_geometry '{self.data_collection_geometry}'. "
                "Must be 'continuous rotation' or 'precession'.")
        
        #apply z rotation to orientations to correct for rotation_axis position
        #convert mosaicity degrees into number of frames to performe moving average during intensity calculation
        degrees_per_frame = (self.integration_semiangle * 2 / self.rocking_curve_sampling)
        self.mosaicity_num_frames = np.round(self.mosaicity / degrees_per_frame).astype(int)
    
    def __len__(self):
        return len(self.rotations)

    def __getitem__(self, idx):
        """
        Returns:
            idx: index of rotation
            intensities: torch.tensor of intensities
        """
        # TODO ideally this should give back the intensities for the rotation idx instead
        # of requiring the model to do this 
        return idx, self.rotations[idx], self.alphas[idx], self.thicknesses[idx] #, self.exp_info[idx.item()+1]
    
    def get_reference_data(self):
        """check for reference data in cif file and load is present.
        There will be reference data if this is a synhtetic dataset.
        Reference data loaded here will be used for rmsd calculation during optim"""
        #TODO: currently use multiple cif parsers, decide on one, probably gemmis is best
        doc = cif.read_file(self.ref_intensities_path)
        block = doc.sole_block()
        atom_site_labels = block.find_values('_atom_site_label')
        
        if not atom_site_labels:
            return None
        else:
            # Load the fractional coordinates
            atom_cols = [
                '_atom_site_label',
                '_atom_site_type_symbol',
                '_atom_site_fract_x',
                '_atom_site_fract_y',
                '_atom_site_fract_z'
            ]

            # Verify all necessary columns exist
            missing_cols = [col for col in atom_cols if block.find_values(col) is None]
            if missing_cols:
                raise ValueError(f"Missing required columns in CIF file: {', '.join(missing_cols)}")

            # Extract the fractional coordinates and optionally other data
            labels = block.find_values('_atom_site_label')
            x_coords = block.find_values('_atom_site_fract_x')
            y_coords = block.find_values('_atom_site_fract_y')
            z_coords = block.find_values('_atom_site_fract_z')
            # Combine the fractional coordinates into a PyTorch tensor
            reference_positions = torch.tensor(
                [
                    [float(x), float(y), float(z)]
                    for x, y, z in zip(x_coords, y_coords, z_coords)
                ],
                dtype=torch.float32
            )  
            return reference_positions
              
    def rotation_axis_correction(self):
        """
        Apply the z-axis rotation and its inverse to each orientation in rocking_curve_orientations.
        This is (I believe) what needs to be done when the pets file has a non-zero rotation angle
        """
        # Bring rotation axis back to 0 for simulation
        z_inverse = rotation_matrix_z(-self.rotation_axis_position)
        #print(f'self.rotation_axis_position:{self.rotation_axis_position}')

        # Rotate each of the existing rotation matrices
        rotated_orientations = []
        for rot_matrix in self.rotations:
            # apply inverse of z rotation to correct for rotation axis position
            combined_rotation = np.dot(z_inverse, rot_matrix)
            rotated_orientations.append(combined_rotation)
        print(f'correcting rotation axis position by {self.rotation_axis_position} degrees')
        # Update the rocking_curve_orientations with the rotated matrices
        self.rotations = rotated_orientations

class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
    """

    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

def get_dataloaders(cfg, default_thickness, optim_orientations_path=None, optim_thicknesses_path=None, fabric=None):
    """
    Returns:
        train_dataloader: torch.utils.data.DataLoader
        val_dataloader: torch.utils.data.DataLoader
    """

    rotations, alphas = generate_rotations(cfg.data.pets_path)
    dataset = RotationDataset(cfg, rotations, alphas, default_thickness, optim_orientations_path, optim_thicknesses_path)

    train_indices = [
        i for i in range(len(dataset)) if i not in cfg.dataloader.ignore_orientations
    ]
    # random subsample using num rotations
    if cfg.dataloader.num_rotations < len(train_indices):
        train_indices = np.random.choice(
            train_indices, cfg.dataloader.num_rotations, replace=False
        )
    # split into train and val
    if cfg.dataloader.val_prop == 1.0:
        val_indices = [i for i in train_indices]
    else:
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=cfg.dataloader.val_prop,
        )

    if cfg.dataloader.rotation_integration_method == "random":
        # subset random sampler ensures idx corresponds to the correct rotation for little s indexing
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    elif cfg.dataloader.rotation_integration_method == "sequential":
        train_sampler = SubsetSequentialSampler(train_indices)
        val_sampler = SubsetSequentialSampler(val_indices)
    elif cfg.dataloader.rotation_integration_method == "single":
        train_sampler = torch.utils.data.SubsetRandomSampler([0])
        val_sampler = torch.utils.data.SubsetRandomSampler([0])
    else:
        # TODO other types of sampling of batches e.g. weighting, or orthogonal rotations in a batch
        # https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
        raise NotImplementedError(
            f"{cfg.dataloader.rotation_integration_method} not implemented"
        )

    # dataloaders
    train_dataloader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=cfg.dataloader.batch_size // fabric.world_size
        if fabric
        else cfg.dataloader.batch_size,
    )
    val_dataloader = DataLoader(
        dataset,
        sampler=val_sampler,
        batch_size=cfg.dataloader.batch_size // fabric.world_size
        if fabric
        else cfg.dataloader.batch_size,
    )

    return train_dataloader, val_dataloader

def generate_rotations(datasets_path):
    # TODO, consolidate/tidy up generate rotation, process_file and generate_crystal_orientations
    # needlessly comfusing. Was written to handle multiple files based as a dir and whould retain
    # this functionality but could be simplified
    all_rotations = []
    all_alphas = []
    # Check if datasets_path is a directory or a single file
    if os.path.isdir(datasets_path):
        # Process each file in the directory
        for fname in sorted(os.listdir(datasets_path)):
            path = os.path.join(datasets_path, fname)
            rotations, alphas = process_file(path)
            # print(len(rotations))
            all_rotations.extend(rotations)
            all_alphas.extend(alphas)
    elif os.path.isfile(datasets_path):
        # Process the single file
        rotations, alphas = process_file(datasets_path)
        all_rotations.extend(rotations)
        all_alphas.extend(alphas)
    else:
        raise ValueError(f"Invalid path: {datasets_path}")

    return all_rotations, all_alphas

def process_file(path):
    rot_u_matrix_file = load_data(path)
    u_matrix = generate_u_matrix(rot_u_matrix_file)
    rotations, alphas = generate_crystal_orientations(
        rot_u_matrix_file, u_matrix
    )
    return rotations, alphas

def generate_crystal_orientations(cif_file, U):
    orientations = []
    alphas = []
    for i in range(len(cif_file["pets"]["_diffrn_zone_axis_id"])):
        alpha = float(cif_file["pets"]["_diffrn_zone_axis_alpha"][i])
        beta = float(cif_file["pets"]["_diffrn_zone_axis_beta"][i])
        omega = float(cif_file["pets"]["_diffrn_zone_axis_omega"][i])
        rot_mat = construct_rotation_matrix(alpha, beta, omega)
        orientation = np.dot(rot_mat, U)
        orientations.append(orientation)
        alphas.append(alpha)
    return orientations, alphas

def generate_u_matrix(cif_file):

    ub = generate_ub_matrix(cif_file)
    Binv = generate_Binv(cif_file=cif_file)
    U = np.matmul(ub, Binv)

    return U

def generate_ub_matrix(cif_file):
    # print(cif_file.keys())
    ub_11 = float(cif_file["pets"]["_diffrn_orient_matrix_UB_11"])
    ub_12 = float(cif_file["pets"]["_diffrn_orient_matrix_UB_12"])
    ub_13 = float(cif_file["pets"]["_diffrn_orient_matrix_UB_13"])
    ub_21 = float(cif_file["pets"]["_diffrn_orient_matrix_UB_21"])
    ub_22 = float(cif_file["pets"]["_diffrn_orient_matrix_UB_22"])
    ub_23 = float(cif_file["pets"]["_diffrn_orient_matrix_UB_23"])
    ub_31 = float(cif_file["pets"]["_diffrn_orient_matrix_UB_31"])
    ub_32 = float(cif_file["pets"]["_diffrn_orient_matrix_UB_32"])
    ub_33 = float(cif_file["pets"]["_diffrn_orient_matrix_UB_33"])

    ub = np.array([[ub_11, ub_12, ub_13], [ub_21, ub_22, ub_23], [ub_31, ub_32, ub_33]])

    return ub

def generate_Binv(cif_file):
    a = float(cif_file["pets"]["_cell_length_a"])
    b = float(cif_file["pets"]["_cell_length_b"])
    c = float(cif_file["pets"]["_cell_length_c"])
    alpha = float(cif_file["pets"]["_cell_angle_alpha"])
    beta = float(cif_file["pets"]["_cell_angle_beta"])
    gamma = float(cif_file["pets"]["_cell_angle_gamma"])
    V = float(cif_file["pets"]["_cell_volume"])
    angles = np.deg2rad([alpha, beta, gamma])
    ca, cb, cc = np.cos(angles)
    sa, sb, sc = np.sin(angles)
    B = np.array(
        [
            [1 / a, 0, 0],
            [-cc / (a * sc), 1 / (b * sc), 0],
            [
                b * c / V * (cc * (ca - cb * cc) / sc - cb * sc),
                a * c / (V * sc) * (ca - cb * cc),
                a * b * sc / V,
            ],
        ]
    )
    B_inv = np.linalg.inv(B)
    return B_inv

def get_exp_ints(exp_path):
    exp_data_dict = {}
    rotation_index_offset = 0

    def process_file(file_path, rotation_index_offset):
        exp_info = load_data(file_path)
        max_index = 0
        for i in range(len(exp_info["pets"]["_refln_zone_axis_id"])):
            index = (
                int(exp_info["pets"]["_refln_zone_axis_id"][i]) + rotation_index_offset
            )
            if index not in exp_data_dict:
                exp_data_dict[index] = []
            h = int(exp_info["pets"]["_refln_index_h"][i])
            k = int(exp_info["pets"]["_refln_index_k"][i])
            l = int(exp_info["pets"]["_refln_index_l"][i])
            hkl = (h, k, l)
            intensity = float(exp_info["pets"]["_refln_intensity_meas"][i])
            sigma = float(exp_info["pets"]["_refln_intensity_sigma"][i])
            exp_data_dict[index].append(
                {"hkl": hkl, "intensity": intensity, "sigma": sigma}
            )
            # Track the maximum index in this file
            if index > max_index:
                max_index = index
        return max_index

    if os.path.isdir(exp_path):
        # Process each file in the directory
        for fname in sorted(os.listdir(exp_path)):
            file_path = os.path.join(exp_path, fname)
            rotation_index_offset = process_file(file_path, rotation_index_offset)
    elif os.path.isfile(exp_path):
        # Process the single file
        process_file(exp_path, rotation_index_offset)
    else:
        raise ValueError(f"Invalid path: {exp_path}")

    return exp_data_dict

def rotation_matrix_z(omega):
    """Create a rotation matrix around the z-axis."""
    omega = np.deg2rad(omega)
    return np.array(
        [
            [np.cos(omega), -np.sin(omega), 0],
            [np.sin(omega), np.cos(omega), 0],
            [0, 0, 1],
        ]
    )

def rotation_matrix_x(alpha):
    """Create a rotation matrix around the x-axis."""
    alpha = np.deg2rad(alpha)
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )

def rotation_matrix_y(beta):
    """Create a rotation matrix around the y-axis."""
    beta = np.deg2rad(beta)
    return np.array(
        [
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)],
        ]
    )

def construct_rotation_matrix(alpha, beta, omega):
    """Construct a full rotation matrix from rotations around z, x, and y axes."""
    R_z = rotation_matrix_z(omega)
    R_x = rotation_matrix_x(alpha)
    R_y = rotation_matrix_y(beta)

    # Combined rotation matrix
    Rot_mat = np.dot(np.dot(R_z, R_x), R_y)

    return Rot_mat

def extract_data_params(pets_path):
    for block in parse_cif(pets_path):
        # Access the measurement details
        measurement_details = block.get('_diffrn_measurement_details', '')
        
        # Initialize variables
        rotation_axis_position = None
        rc_width = None
        mosaicity = None
        dstarmax = None
        
        # Parse each line to find the desired values
        for line in measurement_details.split('\n'):
            if 'rotation axis position' in line:
                rotation_axis_position = float(line.split(':')[1].strip())
            elif 'data collection geometry' in line:
                data_collection_geometry = line.split(':')[1].strip()
            elif 'RC width' in line:
                rc_width = float(line.split(':')[1].strip())
            elif 'mosaicity' in line:
                mosaicity = float(line.split(':')[1].strip())
            elif 'dstarmax' in line:
                dstarmax = float(line.split(':')[1].strip())
        
        return rotation_axis_position, rc_width, mosaicity, dstarmax, data_collection_geometry

def generate_integration_rotation_matrices(semi_angle, num_steps):
    """
    This function generates a series of rotation matrices about x that describe the
    orientations observed during a single virtual frame. We generate in x because these
    matrices are applied to the unit cell after rotation of the unti cell into 
    the pets2 coordinate frame. In the pets2 coordinate frame, the goniometer axis is x.
    if the rotation axis does not equal zero then a further rotation is applied to these orientations
    """
    angles = np.linspace(-semi_angle, semi_angle, num_steps)
    rotation_matrices = []
    for angle in angles:
        # Generate the rotation matrix for each angle
        rot_matrix = rotation_matrix_x(angle)
        rotation_matrices.append(rot_matrix)
    return rotation_matrices

def generate_precession_rotations(precession_angle_deg, Nphi):
    """
    Generate rotation matrices for a precession electron diffraction simulation.

    Parameters:
    -----------
    precession_angle_deg : float
        Precession angle in degrees.
    Nphi : int
        Number of azimuthal rotation steps around the z-axis.

    Returns:
        rotations (list[np.ndarray]): List of 3x3 rotation matrices.
    """
    phi_angles = np.linspace(0, 360, Nphi, endpoint=False)

    rotations = []
    for phi in phi_angles:
        # Rotate around z by phi, then tilt by precession angle around x, then rotate back by -phi around z
        R = rotation_matrix_z(phi) @ rotation_matrix_x(precession_angle_deg) @ rotation_matrix_z(-phi)
        rotations.append(R)

    return rotations


