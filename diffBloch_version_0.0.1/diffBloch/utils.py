from __future__ import annotations

from tqdm import tqdm
import shutil
import os
import itertools
from typing import Optional, Sequence
import pandas as pd
import torch
import numpy as np
import pandas as pd
from ase.cell import Cell
from numba import njit  # type: ignore

from abtem.core.backend import cp
from abtem.core.energy import energy2wavelength, energy2sigma
from abtem.core.constants import kappa



from diffBloch.metrics import get_loss

import wandb

from ast import literal_eval


def reciprocal_cell(cell: np.ndarray | Cell) -> np.ndarray:
    """
    Calculate the reciprocal cell of a unit cell.

    Parameters
    ----------
    cell : 3x3 np.ndarray
        The unit cell.

    Returns
    -------
    3x3 np.ndarray
        The reciprocal cell.
    """
    return np.linalg.pinv(cell).transpose()


def calculate_g_vec(hkl: np.ndarray, cell: np.ndarray | Cell) -> np.ndarray:
    return hkl @ reciprocal_cell(cell)


def calculate_g_vec_length(hkl: np.ndarray, cell: np.ndarray | Cell) -> np.ndarray:
    return np.linalg.norm(calculate_g_vec(hkl, cell), axis=-1)


def hkl_strings_to_array(hkl: list[str]) -> np.ndarray:
    return np.array([tuple(map(int, hkli.split(" "))) for hkli in hkl])


def generate_linear_combinations(
    vectors: np.ndarray, coefficients: Sequence[int], exclude_zero: bool = False
) -> np.ndarray:
    """
    Generate all possible linear combinations of the given vectors with the given
    coefficients.

    Parameters
    ----------
    vectors : np.array
        Array of vectors.
    coefficients : sequence of int
        Coefficients to use in the linear combinations.
    exclude_zero : bool, optional
        Whether to exclude the zero vector from the output.

    Returns
    -------
    np.array
        Array of linear combinations.
    """
    combinations = np.array(
        [
            sum(c * v for c, v in zip(coef_comb, vectors))
            for coef_comb in itertools.product(coefficients, repeat=len(vectors))
        ]
    )
    if exclude_zero:
        combinations = combinations[(combinations == 0).all(axis=1) == 0]
    return combinations


def get_shortest_g_vec_length(cell: Cell) -> float:
    """
    Get the length of the shortest reciprocal space vector in the given unit cell.

    Parameters
    ----------
    cell : Cell
        Unit cell.

    Returns
    -------
    float
        Length of the shortest reciprocal space vector [1/Å].
    """
    coefficients = [-1, 0, 1]
    reciprocal_cell = np.array(cell.reciprocal())
    combinations = generate_linear_combinations(
        reciprocal_cell, coefficients, exclude_zero=True
    )
    return np.min(np.linalg.norm(combinations, axis=1))

def reciprocal_space_gpts(
    cell: np.ndarray | Cell,
    g_max: float,
) -> tuple[int, int, int]:
    # if isinstance(g_max, Number):
    #    g_max = (g_max,) * 3

    # assert len(g_max) == 3

    dk = np.linalg.norm(reciprocal_cell(cell), axis=1)

    gpts = (
        int(np.ceil(g_max / dk[0])) * 2 + 1,
        int(np.ceil(g_max / dk[1])) * 2 + 1,
        int(np.ceil(g_max / dk[2])) * 2 + 1,
    )
    return gpts


def make_hkl_grid(
    cell: np.ndarray | Cell,
    g_max: float,
    axes: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    gpts = reciprocal_space_gpts(cell, g_max)

    freqs = tuple(np.fft.fftfreq(n, d=1 / n).astype(int) for n in gpts)

    freqs = tuple(freqs[axis] for axis in axes)

    hkl_grids = np.meshgrid(*freqs, indexing="ij")
    hkl = np.stack(hkl_grids, axis=-1)

    hkl = hkl.reshape((-1, len(axes)))
    g_vec = calculate_g_vec(hkl, cell)
    hkl = hkl[(g_vec**2).sum(-1) <= g_max**2]
    return hkl


def excitation_errors(g: np.ndarray, energy: float, U0: float = 0.0) -> np.ndarray:
    """
    Calculate excitation error Sg for each reciprocal lattice vector g,
    using the Spence and Zuo method with a mean inner potential correction.

    Parameters
    ----------
    g : np.ndarray
        Reciprocal lattice vectors of shape (N, 3), in units of 1/Å.
    energy : float
        Electron beam energy in eV.
    U0 : float, optional
        Mean inner potential of the crystal in volts (default is 0.0).

    Returns
    -------
    np.ndarray
        Excitation errors Sg of shape (N,), in units of 1/Å.
    """

    l = energy2wavelength(energy)  # in Å
    K0 = 1 / l                     # vacuum wavevector magnitude (1/Å)
    Kmag = np.sqrt(K0**2 + U0)     # corrected wavevector magnitude (1/Å)
    K = np.array([0.0, 0.0, -Kmag])  # beam along -z

    Sg = (np.linalg.norm(K)**2 - np.linalg.norm(K + g, axis=1)**2) / (2 * Kmag)
    return Sg


def get_reflection_condition(hkl: np.ndarray, centering: str):
    """
    Returns a boolean mask indicating which reflections satisfy the reflection condition
    based on the given lattice centering.

    Parameters
    ----------
    hkl : np.ndarray
        Array of shape (N, 3) representing the Miller indices of reflections.
    centering : str
        The lattice centering type. Must be one of "P", "I", "F", "A", "B", or "C".

    Returns
    -------
    np.ndarray
        Boolean mask indicating which reflections satisfy the reflection condition.
    """
    if centering.lower() == "f":
        all_even = (hkl % 2 == 0).all(axis=1)
        all_odd = (hkl % 2 == 1).all(axis=1)
        return all_even + all_odd
    elif centering.lower() == "i":
        return hkl.sum(axis=1) % 2 == 0
    elif centering.lower() == "a":
        return (hkl[1:].sum(axis=1) % 2 == 0).all(axis=1)
    elif centering.lower() == "b":
        return (hkl[:, [0, 1]].sum(axis=1) % 2 == 0).all(axis=1)
    elif centering.lower() == "c":
        return (hkl[:-1].sum(axis=1) % 2 == 0).all(axis=1)
    elif centering.lower() == "p":
        return np.ones(len(hkl), dtype=bool)
    else:
        raise ValueError()


@njit(nogil=True, error_model="numpy")
def fast_filter_excitation_errors(mask, g, orientation_matrices, wavelength, sg_max):
    g_length_2 = (g**2).sum(axis=-1)

    b = 0.5 * wavelength * g_length_2
    for i in range(len(orientation_matrices)):
        R = orientation_matrices[i]

        sg = -g[:, 0] * R[2, 0] - g[:, 1] * R[2, 1] - g[:, 2] * R[2, 2] - b

        mask += np.abs(sg) < sg_max


def filter_reciprocal_space_vectors(
    hkl: np.ndarray,
    reciprocal_cell: Cell,
    energy: float,
    sg_max: float,
    g_max: float,
    centering: str = "P",
    orientation_matrices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Filter reciprocal space vectors based on excitation errors and reflection
    conditions.

    Parameters
    ----------
    hkl : np.ndarray
        Reciprocal space vectors.
    cell : Cell
        Unit cell.
    energy : float
        Electron energy [eV].
    sg_max : float
        Maximum excitation error [1/Å].
    g_max : float
        Maximum scattering vector length [1/Å].
    centering : str, optional
        Crystal centering must be one of 'P', 'I', 'A', 'B', 'C' or 'F'. Default is 'P'.
    orientation_matrices : np.ndarray, optional
        Orientation matrices for each crystallographic direction.

    Returns
    -------
    np.ndarray
        Mask for the reciprocal space vectors.
    """
    g = hkl @ reciprocal_cell
    g_length = np.linalg.norm(g, axis=-1)
    test_g = np.array([[0,-6,0], [3,-6,0], [3, -3, 0], [-3,0,0]]) @ reciprocal_cell
    test = np.linalg.norm(test_g, axis = 1) 
    if orientation_matrices is None:
        mask = np.abs(excitation_errors(g, energy)) <= sg_max
    mask *= get_reflection_condition(hkl, centering)

    mask *= g_length <= g_max

    return mask

def raveled_hkl_to_hkl_torch(
    array: torch.Tensor,
    hkl_source: np.ndarray,
    hkl_destination: np.ndarray,
    gpts: tuple[int, int, int],
) -> torch.Tensor:
    """
    Convert a raveled array to a 3D array with the shape of the structure factor using PyTorch,
    while maintaining gradient tracking.
    
    Parameters
    ----------
    array : torch.Tensor
        The raveled array (requires_grad=True).
    hkl_source : np.ndarray
        The reciprocal space vectors as Miller indices for the source array.
    hkl_destination : np.ndarray
        The reciprocal space vectors as Miller indices for the destination array.
    gpts : tuple of ints
        The number of grid points in the 3D structure factor.

    Returns
    -------
    torch.Tensor
        The 3D array.
    """

    # Use the NumPy ravel_hkl function to ravel the hkl_source and hkl_destination
    hkl_source_raveled = ravel_hkl(hkl_source, gpts)
    hkl_destination_raveled = ravel_hkl(hkl_destination, gpts)

    # Convert hkl raveled arrays back to PyTorch tensors
    hkl_source_raveled = torch.tensor(hkl_source_raveled, dtype=torch.long, device=array.device)
    hkl_destination_raveled = torch.tensor(hkl_destination_raveled, dtype=torch.long, device=array.device)

    # Create an index tensor that maps hkl_source_raveled to array
    # We use torch.gather instead of a dictionary lookup to maintain gradients.
    max_index = hkl_source_raveled.max() + 1  # Find the maximum index for sparse tensor creation

    # Create a sparse tensor with the values of array at hkl_source_raveled locations
    sparse_array = torch.zeros(max_index, dtype=array.dtype, device=array.device)
    sparse_array.index_add_(0, hkl_source_raveled, array)

    # Now use hkl_destination_raveled to get the corresponding values
    output_array = sparse_array[hkl_destination_raveled]

    return output_array


def ravel_hkl(hkl: np.ndarray, gpts: tuple[int, int, int]) -> np.ndarray:
    hkl = np.asarray(hkl)
    shift = np.array((gpts[0] // 2, gpts[1] // 2, gpts[2] // 2))
    hkl = hkl + shift
    multi_index = (hkl[..., 0], hkl[..., 1], hkl[..., 2])
    return np.ravel_multi_index(multi_index, gpts)


def fill_diagonal_torch(A: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
    """
    Fills the diagonal of a 2D square tensor A with the values from diag.
    
    Parameters
    ----------
    A : torch.Tensor
        A square 2D tensor whose diagonal needs to be filled.
    diag : torch.Tensor
        A 1D tensor containing the values to fill the diagonal of A.
    
    Returns
    -------
    torch.Tensor
        The tensor A with its diagonal replaced by diag values.
    """
    
    # Ensure that A is a square matrix and diag has appropriate length
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert len(diag) == A.shape[0], "diag must have the same length as the number of rows in A"
    
    # Fill the diagonal using advanced indexing
    A_copy = A.clone()  # Clone the tensor to avoid in-place modification
    indices = torch.arange(A.shape[0], device=A.device)
    A_copy[indices, indices] = diag
    
    return A_copy

def initialize_scaling_factor(
    exp_intensities, simulated_intensities, sigmas, r_value_method="rbragg_abs"
    ):
    """
    Initialize scaling factor to minimize R value
    Params:
    - scaling_factor: scaling factor for each rotation
    - exp_intensities: experimental intensities
    - simulated_intensities: simulated intensities
    - sigmas: experimental sigmas
    - r_value_method: whether to use wRbragg, rbragg_abs, or rbragg_squared
    Returns:
    - scaling_factor: scaling factor for each rotation
    """
    start = 0.02
    end = 2
    num_points = 100
    grid_points = torch.linspace(start, end, num_points, device=exp_intensities.device)
    exp_sum = torch.sum(exp_intensities)
    sim_sum = torch.sum(simulated_intensities)
    scaled_intensities = (
        grid_points.view(-1, 1) * (exp_sum / sim_sum) * simulated_intensities
    )
    losses = get_loss(r_value_method)(scaled_intensities, exp_intensities, sigmas)
    min_value, min_idx = torch.min(losses, dim=0)
    scaling_factor = grid_points[min_idx]

    return scaling_factor, min_value

def velocity(energy):
    v = 3 * 10**8 * np.sqrt(1 - (1 + 1.96e-6 * energy)**-2)
    return v

def relativistic_constant(v):
    gamma = 1 / np.sqrt(1 - (v**2 / ((3 * 10**8))**2))
    return gamma

def convert_Fgb_to_Fgxray(bloch_nn, Fgb=None):
    """
    following methodology of J.C.H. Spence Acta Cryst. (1993). A49, 231-260
    upto equation 22
    
    Convert Fgb/cell_volume, which is our structure factor to Ug, then to Fxray
    """
    # get the cell volume in Å^3
    cell_volume = bloch_nn.structure_factor_net.atoms.cell_volume()
    reciprocal_cell = bloch_nn.structure_factor_net.atoms.reciprocal_cell()
    # V_g in units of V/Å^3, convert to V
    if Fgb is None:
        print("Fgb is None, using bloch_nn.Fgb)")
        Fgb = bloch_nn.Fgb
    Fgb = torch.conj(Fgb)
    # convert Fgb/cell_volume to energy dependent electron structure factor U_g (units of Å^-2)
    prefactor = energy2sigma(bloch_nn.energy) / (kappa * energy2wavelength(bloch_nn.energy) * np.pi)
    U_g = Fgb * prefactor
    # provided that s, the cell volume and U_g are in Å units, then C is  
    C = 131.2625

    # need to get the expanded unit cell 
    expanded_positions, expanded_atomic_numbers, expanded_disps, occupancy = bloch_nn.structure_factor_net.atoms()

    #setting up summation in eq 22 in Spence
    Z = torch.tensor(expanded_atomic_numbers, device=U_g.device)
    hkl = bloch_nn.structure_factor_net.hkl
    g_mag = bloch_nn.structure_factor_net.g_vec_length
    #I'm assuming the   relativistic constant he uses is the lorentz factor
    v = velocity(bloch_nn.energy)
    gamma = relativistic_constant(v)

    # handles isotropic and anisotropic displacement factors
    exp_dwf = bloch_nn.structure_factor_net.calculate_dwf_factor(expanded_disps, hkl, reciprocal_cell)
    exp_phase = torch.exp(-2.0j * torch.pi * torch.matmul(expanded_positions, hkl.T))

    Z_term = Z.unsqueeze(0).T * exp_dwf

    result_per_hkl = Z_term * exp_phase
    summed_result_per_hkl = torch.sum(result_per_hkl, dim = 0)
    second_term = torch.tensor((C * cell_volume * (g_mag / 2)**2 / gamma), device = U_g.device) * U_g
    Fgxray_unmasked = summed_result_per_hkl - second_term
    threshold = 1e-12
    mask_real = (torch.abs(Fgxray_unmasked.real) >= threshold)
    mask_imag = (torch.abs(Fgxray_unmasked.imag) >= threshold)

    # Use torch.where to selectively zero out small values
    new_real = torch.where(mask_real, Fgxray_unmasked.real, torch.zeros_like(Fgxray_unmasked.real))
    new_imag = torch.where(mask_imag, Fgxray_unmasked.imag, torch.zeros_like(Fgxray_unmasked.imag))
    Fg = torch.complex(new_real, new_imag)

    return Fg

def convert_Fg_to_Ug(bloch_nn, Fg =None, return_Fgb=False, hkls = None):
    
    # get the cell volume in Å^3
    cell_volume = bloch_nn.structure_factor_net.atoms.cell_volume()
    reciprocal_cell = bloch_nn.structure_factor_net.atoms.reciprocal_cell()
    C = 131.2625
    hkls = torch.tensor(hkls, device = bloch_nn.device, dtype = torch.float64)
    # need to get the expanded unit cell 
    expanded_positions, expanded_atomic_numbers, expanded_disps, occupancy = bloch_nn.structure_factor_net.atoms()
    Z = torch.tensor(expanded_atomic_numbers, device=bloch_nn.device)
    gvec = hkls @ reciprocal_cell
    g_mag = np.linalg.norm(gvec, axis=1)

    v = velocity(bloch_nn.energy)
    gamma = relativistic_constant(v)
    print(f'gamma: {gamma}')

    exp_dwf = bloch_nn.structure_factor_net.calculate_dwf_factor(expanded_disps, hkls, reciprocal_cell)

    exp_phase = torch.exp(-2.0j * torch.pi * torch.matmul(expanded_positions, hkls.T))

    Z_term = Z.unsqueeze(0).T * exp_dwf 

    result_per_hkl = Z_term * exp_phase
    summed_result_per_hkl = torch.sum(result_per_hkl, dim = 0)
    scalar = torch.tensor(((C * cell_volume * (g_mag / 2)**2) / gamma), device = bloch_nn.device) 
    electron_structure_factors = (summed_result_per_hkl.detach() - Fg) / scalar

    #returns electron structure factors within the context of the Born approximation / the unit cell. 
    #This seems to be form of choice in abtem so sticking with it in our code
    if return_Fgb:
        prefactor = energy2sigma(bloch_nn.energy) / (kappa * energy2wavelength(bloch_nn.energy) * np.pi)
        electron_structure_factors = electron_structure_factors / prefactor
    
    #numerical precision
    threshold = 1e-12
    mask_real = (torch.abs(electron_structure_factors.real) >= threshold)
    mask_imag = (torch.abs(electron_structure_factors.imag) >= threshold)

    # Use torch.where to selectively zero out small values
    new_real = torch.where(mask_real, electron_structure_factors.real, torch.zeros_like(electron_structure_factors.real))
    new_imag = torch.where(mask_imag, electron_structure_factors.imag, torch.zeros_like(electron_structure_factors.imag))
    electron_structure_factors = torch.complex(new_real, new_imag)
    
    return electron_structure_factors

def structure_factor_to_density(structure_factors, grid_size, hkls):
    """
    Convert structure factors to a density map (electrostatic potential or electron density).
    
    Parameters:
    - structure_factors (torch.Tensor or np.ndarray): Array of complex structure factors (V_g or F_g), 
      where each entry corresponds to an (hkl) reflection. These values should be in units of V/Å^3 
      for electrostatic potential or e/Å^3 for electron density.
    - grid_size (tuple of int): The size of the grid (nx, ny, nz) for the real-space density map.
    - hkls (np.ndarray or list): Array or list of (h, k, l) Miller indices corresponding to the 
      structure factors.

    Returns:
    - density (np.ndarray): Real part of the electron density or electrostatic potential map, 
      computed on a grid of size `grid_size`.
      
    Assumptions:
    - If V_g (electrostatic potential structure factors) are provided, the function returns 
      the electrostatic potential map.
    - If F_g (electron density structure factors) are provided, the function returns the 
      electron density map.
      
    The function assumes that the structure factors are given in units of V/Å^3 or e/Å^3 and 
    applies a phase factor based on the (hkl) reflections to compute the density in real space.
    """
    
    density = np.zeros(grid_size, dtype=np.complex128)
    
    x = np.linspace(0, 1, grid_size[0])
    y = np.linspace(0, 1, grid_size[1])
    z = np.linspace(0, 1, grid_size[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    structure_factors = structure_factors.detach().cpu().numpy() if hasattr(structure_factors, 'detach') else structure_factors

    for hkl, sf_hkl in zip(hkls, structure_factors):
        h, k, l = hkl
        phase_factor = np.exp(-2j * np.pi * (h*X + k*Y + l*Z))
        density += sf_hkl * phase_factor
    
    density = np.real(density)
    return density

def load_checkpoint(
    model,
    optimizer,
    scheduler=None,
    device=None,
    checkpoint_file: str = "asu.pth.tar",
    fabric=None,
    load_optim=True,
    load_structure=True,
):
    """Loads a model checkpoint.
    Params:
    - model (nn.Module): initialised model
    - optimizer (nn.optim): initialised optimizer
    - scheduler (nn.optim.lr_scheduler): initialised scheduler
    - device (torch.device): device model is on
    - checkpoint_file: file to load checkpoint from
    - fabric
    - load_optim: whether to load optimizer/scheduler state dict
    Returns:
    - model with loaded state dict
    - optimizer with loaded state dict
    - scheduler with loaded state dict
    - epoch (int): epoch checkpoint was saved at
    - loss (float): loss at checkpoint
    """
    if fabric:
        print(f" Loading checkpoint from: {checkpoint_file}")

        checkpoint = fabric.load(checkpoint_file)
    else:
        print(f" Loading checkpoint from: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device or "cpu")
        print(f"Checkpoint was saved at epoch {checkpoint.get('epoch', 'unknown')}, loss: {checkpoint.get('loss', 'N/A')}")

    full_state_dict = checkpoint["state_dict"]

    if not load_structure:
        # Keep only thickness_nn parameters
        filtered_state_dict = {
            k: v for k, v in full_state_dict.items()
            if "thickness_nn" in k
        }
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(full_state_dict)

    if load_optim:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
    print(
        f"Loaded {checkpoint_file}, "
        f"trained to epoch {checkpoint['epoch']} with rbragg {checkpoint['rbragg']}, loss {checkpoint['loss']}"
        f"optimizer state loaded: {load_optim}"
    )

    return model, optimizer, scheduler, checkpoint["epoch"], checkpoint["loss"]


def save_checkpoint(
    checkpoint_dict: dict, file_name: str = "asu", is_best: bool = False
):
    """Saves a model checkpoint to file. Keeps most recent and best model.
    Params:
    - checkpoint_dict (dict): dict containing all model state info, to pickle
    - is_best (bool): whether this checkpoint is the best seen so far.
    """
    # files for checkpoints
    checkpoint_file = os.path.join(wandb.run.dir, f"{file_name}.pth.tar")
    best_file = os.path.join(wandb.run.dir, f"{file_name}_best.pth.tar")
    torch.save(checkpoint_dict, checkpoint_file)
    wandb.save(checkpoint_file, policy="live")  # save to wandb
    print(f"Saved checkpoint to {checkpoint_file}")

    if is_best:
        shutil.copyfile(checkpoint_file, best_file)
        print(f"Saved best checkpoint to {best_file}")
        wandb.save(best_file, policy="live")  # save to wandb

def gmax_mask(
    hkl: np.ndarray,
    reciprocal_cell: np.ndarray,
    g_max: float,
) -> torch.Tensor:
    """
    mask for hkls greater than a defined g_max, used to prevent grad tracking on 
    Fgb greater than a defined g_max
    """
    g = hkl @ reciprocal_cell

    g_norm = np.linalg.norm(g, axis=1)

    # create a mask such that every entry in g is false if bigger than g_max
    mask = g_norm <= g_max
   
    return torch.tensor(mask, dtype=torch.bool)

def resolution_filter_diffraction_intensities(
    experimental_intensities: torch.tensor,
    experimental_sigmas: torch.tensor,
    simulated_intensities: torch.tensor,
    hkl: tuple,
    g_max: float,
    g_min: float,
    reciprocal_lattice_vectors: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Filter diffraction intensities based on resolution limits.

    Parameters
    ----------
    diffraction_intensities : np.ndarray
        The diffraction intensities.
    hkl : np.ndarray
        The reciprocal space vectors as Miller indices.
    g_max : float
        The maximum scattering vector length [1/Å].
    g_min : float
        The minimum scattering vector length [1/Å].
    reciprocal_lattice_vectors : np.ndarray
        The reciprocal lattice vectors.

    Returns
    -------
    np.ndarray
        The filtered diffraction intensities.
    np.ndarray
        The filtered reciprocal space vectors.
    """
    hkl = np.array(hkl)
    g = np.linalg.norm(hkl @ reciprocal_lattice_vectors, axis=1)
    mask = (g >= g_min) & (g <= g_max)
    masked_hkl = hkl[mask]
    masked_hkl = masked_hkl.tolist()
    mask = torch.tensor(mask, device=experimental_intensities.device)
    filtered_simulated_intensities = simulated_intensities[mask]
    return experimental_intensities[mask], experimental_sigmas[mask], filtered_simulated_intensities, masked_hkl

def create_hkl_mask(all_hkls, observed_hkls):
    all_hkls_expanded = all_hkls.unsqueeze(1)
    observed_hkls_expanded = observed_hkls.unsqueeze(0)
    mask = (all_hkls_expanded == observed_hkls_expanded).all(dim = -1)
    mask = mask.any(dim = 1)
    return torch.tensor(mask, dtype = torch.bool)

def check_if_gradients_zero(masked_grad, combined_mask):
    # Check if entries in combined_mask are False
    if not combined_mask.any():
        return True
    # Check if all gradients of Fgb are zero after applying the combined mask

    if torch.all(masked_grad.real == 0) and torch.all(masked_grad.imag == 0):
        return True
    
    return False


def mask_structure_factors(
    array: torch.Tensor,
    hkl_source: np.ndarray,
    hkl_destination: np.ndarray,
) -> torch.Tensor:
    """
    Extract entries from array that correspond to hkl_destination, which is a subset of hkl_source.

    Parameters
    ----------
    structure factors : torch.Tensor
        The array with sf values corresponding to hkl_source (requires_grad=True).
    hkl_source : np.ndarray
        The reciprocal space vectors as Miller indices for the source array.
    hkl_destination : np.ndarray
        The reciprocal space vectors as Miller indices for the destination structure factor array (a subset of hkl_source).

    Returns
    -------
    torch.Tensor
        The array with entries corresponding to hkl_destination.
    """
    # Convert hkl arrays to torch tensors for compatibility with array's device
    hkl_source_tensor = torch.tensor(hkl_source, dtype=torch.long, device=array.device)
    hkl_destination_tensor = torch.tensor(hkl_destination, dtype=torch.long, device=array.device)
    
    # Create a list to store the indices
    indices = []

    # Loop over each row in hkl_destination and find the matching index in hkl_source
    for hkl in hkl_destination_tensor:
        index = (hkl_source_tensor == hkl).all(dim=1).nonzero(as_tuple=False)
        if index.numel() > 0:
            indices.append(index[0].item())  # Get the first match and append the index

    # Convert the list of indices to a tensor
    indices = torch.tensor(indices, dtype=torch.long, device=array.device)

    # Extract the values from array using the found indices
    output_array = array[indices]
    masked_hkls  = hkl_source_tensor[indices]
    
    return output_array


def create_Fgb_symmetry_restraints(symmetry_operations, hkls, Fgb, mask, tol=1e-4):
    """
    Create symmetry restraints for Fgb. This function calculates the symmetry-related hkls
    Parameters:
    - symmetry_operations: list of symmetry operations (from ase)
    - hkls: list of hkls
    - Fgb: Fgbs tensor
    - mask: mask for hkls (tensor of bools, shape [N,]) - if True, the hkl/Fgb is considered
    - tol: tolerance for checking Fgb magnitudes
    Returns:
    - symmetry_related_idx_list: list of symmetry-related hkls (index pairs)
    - phase_diff_list: list of phase differences between symmetry-related hkl pairs (index 1 - index 0)
    need to do these seperately to allow for indexing (torch int vs float)
    """

    # Apply the mask to only keep relevant hkls and Fgb values
    hkls_masked = hkls[mask]
    Fgb_masked = Fgb[mask]
    
    # Keep track of the original indices after masking
    masked_indices = torch.nonzero(mask).flatten()

    symmetry_related_idx_list = []
    phase_diff_list = []
    Fgb_mag = Fgb_masked.abs()
    seen_pairs = set() # prevent double counting
    print("Calculating symmetry-related hkls")
    for i, hkl in enumerate(tqdm(hkls_masked)):

        for rot, trans in symmetry_operations:
            #transformed_hkl = torch.matmul(torch.inverse(torch.tensor(rot, dtype=torch.float, device=hkl.device)).T, hkl.float())
            transformed_hkl = torch.matmul(torch.tensor(rot, dtype=torch.float, device=hkl.device).T, hkl.float())
            transformed_hkl = torch.round(transformed_hkl).to(torch.int)  # Ensure integer values
            transformed_hkl_tuple = tuple(transformed_hkl.tolist())
            
            # Check if the transformed hkl exists in hkls_masked
            mask_transformed = (hkls_masked == transformed_hkl).all(dim=1)
            if torch.any(mask_transformed):
                transformed_idx = torch.nonzero(mask_transformed)[0].item()
                phase_diff = 1-torch.cos(torch.angle(Fgb_masked[transformed_idx]) - torch.angle(Fgb_masked[i]))
                transformed_Fgb_mag = Fgb_mag[transformed_idx]
                if not torch.isclose(transformed_Fgb_mag, Fgb_mag[i], atol=tol):
                    raise ValueError(f"Mismatch in Fgb magnitude for {hkl.tolist()} and symmetry-related {transformed_hkl_tuple}. "
                                     f"Original |Fgb| = {Fgb_mag[i]}, Symmetry-related |Fgb| = {transformed_Fgb_mag}")

                # Track the original indices using masked_indices
                original_i = masked_indices[i].item()
                original_transformed_idx = masked_indices[transformed_idx].item()
                pair = tuple(sorted([original_i, original_transformed_idx]))
                if pair not in seen_pairs and original_i != original_transformed_idx:  # Avoid identical pairs
                    symmetry_related_idx_list.append([original_i, original_transformed_idx])
                    phase_diff_list.append(phase_diff.item())
                    seen_pairs.add(pair)

                # Now calculate the Friedel pair for the transformed hkl
                friedel_hkl = -transformed_hkl
                friedel_mask = (hkls_masked == friedel_hkl).all(dim=1)
                if torch.any(friedel_mask):
                    friedel_hkl_idx = torch.nonzero(friedel_mask)[0].item()
                    friedel_phase_diff = 1 - torch.cos(torch.angle(torch.conj(Fgb_masked[transformed_idx])) - torch.angle(Fgb_masked[i]))

                    friedel_pair = tuple(sorted([original_i, masked_indices[friedel_hkl_idx].item()]))
                    if friedel_pair not in seen_pairs and original_i != masked_indices[friedel_hkl_idx].item():
                        symmetry_related_idx_list.append([friedel_pair[0], friedel_pair[1]])
                        phase_diff_list.append(friedel_phase_diff.item())
                        seen_pairs.add(friedel_pair)
                else:
                    raise ValueError(f"Friedel pair for {transformed_hkl_tuple} not found in hkls")
            else:
                raise ValueError(f"Symmetry-related {transformed_hkl_tuple} not found in hkls")

    return symmetry_related_idx_list, phase_diff_list



def structure_factor_to_density(structure_factors, grid_size, hkls):
    """
    Convert structure factors to a density map (electrostatic potential or electron density).
    
    Parameters:
    - structure_factors (torch.Tensor or np.ndarray): Array of complex structure factors (V_g or F_g), 
      where each entry corresponds to an (hkl) reflection. These values should be in units of V/Å^3 
      for electrostatic potential or e/Å^3 for electron density.
    - grid_size (tuple of int): The size of the grid (nx, ny, nz) for the real-space density map.
    - hkls (np.ndarray or list): Array or list of (h, k, l) Miller indices corresponding to the 
      structure factors.

    Returns:
    - density (np.ndarray): Real part of the electron density or electrostatic potential map, 
      computed on a grid of size `grid_size`.
      
    Assumptions:
    - If V_g (electrostatic potential structure factors) are provided, the function returns 
      the electrostatic potential map.
    - If F_g (electron density structure factors) are provided, the function returns the 
      electron density map.
      
    The function assumes that the structure factors are given in units of V/Å^3 or e/Å^3 and 
    applies a phase factor based on the (hkl) reflections to compute the density in real space.
    """
    
    density = np.zeros(grid_size, dtype=np.complex128)
    
    x = np.linspace(0, 1, grid_size[0])
    y = np.linspace(0, 1, grid_size[1])
    z = np.linspace(0, 1, grid_size[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    structure_factors = structure_factors.detach().cpu().numpy() if hasattr(structure_factors, 'detach') else structure_factors

    for hkl, sf_hkl in zip(hkls, structure_factors):
        h, k, l = hkl
        phase_factor = np.exp(-2j * np.pi * (h*X + k*Y + l*Z))
        density += sf_hkl * phase_factor
    
    density = np.real(density)
    return density

def save_cube(filename, grid_size, electron_density, unit_cell, origin=(0, 0, 0)):
    with open(filename, 'w') as f:
        # Write the header
        f.write("CUBE file generated by Python\n")
        f.write("Electron density map\n")
        
        # Write origin of volumetric data with 0 atoms
        f.write(f"{0:5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
        
        # Write unit cell and grid info
        for i in range(3):
            f.write(f"{grid_size[i]:5d} {unit_cell[i][0]:12.6f} {unit_cell[i][1]:12.6f} {unit_cell[i][2]:12.6f}\n")
        
        # Write electron density values
        electron_density_flat = electron_density.flatten()
        for i, value in enumerate(electron_density_flat):
            f.write(f"{value:13.5e}")
            if (i + 1) % 6 == 0:
                f.write("\n")
        f.write("\n")

def get_density_difference_maps(intensity_file_path, cell_volume, grid_size):
    """ 
    Calculate the dynamical and kinematical difference electrostatic potential maps.

    This function computes the difference between dynamical and kinematical structure factors and converts them into 
    electrostatic potential difference maps, based on the methodology described in the publication:
    https://www.science.org/doi/10.1126/science.aak9652. The input data is expected to be a CSV file containing 
    experimental and simulated intensities, electron structure factor magnitudes and phases for a range of hkls.

    The procedure involves:
    1. Averaging intensities across reflections with the same hkl indices.
    2. Computing the kinematical and dynamical structure factor differences.
    3. Using these differences to create electrostatic potential difference maps.

    Args:
        intensity_file_path (str): Path to the CSV file containing the following columns:
            - 'hkls': A list of Miller indices [h, k, l] for each reflection.
            - 'exp_intensity': Experimental intensities.
            - 'sim_intensity': Simulated intensities.
            - 'vg_abs': Magnitudes of the structure factors (scaled by 1/cell_volume).
            - 'vg_phase': Phases of the structure factors (in radians).
        cell_volume (float): The volume of the unit cell, used to scale the structure factor amplitudes.
        grid_size (tuple): The size of the 3D grid (in voxels) for the output electrostatic potential maps.

    Returns:
        tuple: Two 3D numpy arrays representing the electrostatic potential difference maps:
            - dyn_difference: The electorstatic potential difference map for dynamical structure factors.
            - kin_difference: The electrostatic potential difference map for kinematical structure factors.

    Notes:
        - The CSV file must be in a specific format, including columns for hkls, experimental intensities, simulated 
          intensities, structure factor magnitudes, and phases.
        - The computed density maps are returned on a grid with the specified size.
        - The 'vg_abs' values are expected to have been scaled by the inverse of the cell volume before being saved 
          in the CSV, so this function rescales them by multiplying with the provided cell volume.
        - The calculation assumes that the cell volume and scaling factor are real and positive.

    """
    
    df = pd.read_csv(intensity_file_path)

    #Get csv file in correct format, average across common hkls
    df['hkls'] = df['hkls'].apply(literal_eval)
    df[['h', 'k', 'l']] = pd.DataFrame(df['hkls'].tolist(), index=df.index)
    df = df.drop(columns=['hkls'])
    df = df.set_index(['h', 'k', 'l'])
    df = df.groupby(['h', 'k', 'l'])
    df = df.mean()

    #extract exp intensities
    exp_intensities = df['exp_intensity'].values
    sim_intensities = df['sim_intensity'].values

    #saved Fgb have been divided by cell volume, 
    #the below operation on the mag only valid if volume (scalar) is real and positive, which it always will be
    model_amplitudes = df['vg_abs'].values * cell_volume
    phase_angle = df['vg_phase'].values

    kin_scale_factor = np.sum(np.sqrt(exp_intensities)) / np.sum(model_amplitudes)

    #get the kinematical and dynamical difference maps
    delta_F_mag_dyn = (np.sqrt(exp_intensities) - np.sqrt(sim_intensities)) * (model_amplitudes / np.sqrt(exp_intensities))
    delta_F_mag_kin = np.sqrt(exp_intensities) - (model_amplitudes * kin_scale_factor)

    delta_F_dyn = delta_F_mag_dyn * (np.cos(phase_angle) + 1j * np.sin(phase_angle))
    delta_F_kin = delta_F_mag_kin * (np.cos(phase_angle) + 1j * np.sin(phase_angle))

    hkls = np.array(df.index.to_list())

    dyn_difference = structure_factor_to_density(delta_F_dyn, grid_size, hkls)
    kin_difference = structure_factor_to_density(delta_F_kin, grid_size, hkls)

    return dyn_difference, kin_difference

def sample_thicknesses(mu, sigma, num_samples):
    """
    softplus to avoid nonnegative thicknesses
    """
    epsilon = torch.randn(num_samples, device = mu.device)
    thicknesses = torch.nn.functional.softplus(mu + sigma * epsilon)
    return thicknesses

def calculate_resolution_dependent_background(
    max_resolution,
    intensities,
    magnitudes,
    background_fraction=0.05,
    epsilon=1e-3
):
    """
    Calculate the resolution-dependent background level with a Gaussian decay.

    Parameters:
    - max_resolution (float): Maximum resolution of the diffraction intensities.
    - intensities (torch.tensor): Diffraction intensities (shape [N] or [t, N]).
    - magnitudes (torch.tensor): Magnitudes of the diffraction intensities (shape [N] or [t, N]).
    - background_fraction (float): Fraction of total diffraction signal for background.
    - epsilon (float): Threshold value where decay reaches near zero.

    Returns:
    - background_level (torch.tensor): Background level for each intensity, decaying with resolution (same shape as `intensities`).
    - sigma (float): Standard deviation (scale) of the Gaussian decay.
    """

    # Calculate sigma based on max_resolution and epsilon (scalar)
    sigma = max_resolution / torch.sqrt(-2 * torch.log(torch.tensor(epsilon, device=intensities.device)))

    # Ensure magnitudes have the same shape as intensities
    if magnitudes.dim() == 1 and intensities.dim() == 2:  
        magnitudes = magnitudes.unsqueeze(0).expand_as(intensities)

    # Compute the Gaussian decay background level
    gaussian_decay = torch.exp(-magnitudes**2 / (2 * sigma**2))

    # Compute background level per reflection
    background_level = background_fraction * intensities * gaussian_decay

    return background_level, sigma


# def calculate_resolution_dependent_background(
#     max_resolution,
#     intensities,
#     magnitudes,
#     background_fraction=0.05,
#     epsilon=1e-3
# ):
#     """
#     Calculate the resolution-dependent background level with a Gaussian decay.

#     Parameters:
#     - max_resolution (float): Maximum resolution of the diffraction intensities.
#     - intensities (torch.tensor): Diffraction intensities (shape [N] or [t, N]).
#     - magnitudes (torch.tensor): Magnitudes of the diffraction intensities (shape [N] or [t, N]).
#     - background_fraction (float): Fraction of total diffraction signal for background.
#     - epsilon (float): Threshold value where decay reaches near zero.

#     Returns:
#     - background_level (torch.tensor): Background level for each intensity, decaying with resolution (same shape as `intensities`).
#     - sigma (torch.tensor): Standard deviation (scale) of the Gaussian decay (same shape as `intensities`).
#     """

#     # Calculate sigma based on max_resolution and epsilon
#     sigma = max_resolution / torch.sqrt(-2 * torch.log(torch.tensor(epsilon, device=intensities.device)))

#     # Ensure magnitudes and intensities have matching shapes
#     if magnitudes.dim() == 1 and intensities.dim() == 2:  # If magnitudes is [N] but intensities is [t, N]
#         magnitudes = magnitudes.unsqueeze(0).expand_as(intensities)  # Expand to [t, N]

#     # Compute the Gaussian decay background level
#     gaussian_decay = torch.exp(-magnitudes**2 / (2 * sigma**2))

#     # Compute total signal per thickness t
#     total_signal = torch.sum(intensities, dim=-1, keepdim=True)  # Shape [t, 1] or [1] if 1D

#     # Compute background level
#     background_level = background_fraction * total_signal * gaussian_decay

#     # Expand sigma if needed
#     sigma = sigma.expand_as(intensities)

#     return background_level, sigma

# def scale_intensities(intensities, hkls, I_max=1e8, global_scale_factor=None):
#     """
#     Scale simulated diffraction intensities to fit within a realistic detector range.
#     Reflections with intensities below 1 count across all thicknesses are removed.
    
#     Parameters:
#     - intensities (torch.tensor): Simulated diffraction intensities (shape [M, N]).
#     - hkls (list or np.array): List of Miller indices (shape [N, 3]).
#     - I_max (float): Maximum detectable intensity before saturation.
#     - global_scale_factor (float, optional): Global scale factor to apply to all intensities and handle per dataset scaling rather than per pattern.

#     Returns:
#     - scaled_intensities (torch.tensor): Rescaled intensities after filtering (shape [M, N_filtered]).
#     - filtered_hkls (np.array): Filtered Miller indices (shape [N_filtered, 3]).
#     """
#     M, N = intensities.shape  # M = thickness count, N = number of reflections

#     # Convert hkls to a NumPy array if it is a list
#     hkls_array = np.array(hkls, dtype=int)

#     # Ensure hkls has the correct shape
#     assert hkls_array.shape == (N, 3), f"Expected hkls shape ({N}, 3), got {hkls_array.shape}"
   
#     # Normalize intensities so the maximum does not exceed I_max
#     if global_scale_factor is not None:
#         scaling_factor = global_scale_factor
#     else:
#         scaling_factor = I_max / torch.max(intensities)
    
#     scaled_intensities = intensities * scaling_factor

#     # Create a mask where a reflection is valid if it has at least 1 count in any thickness
#     valid_mask = (scaled_intensities >= 1).any(dim=0).cpu().numpy()  # Shape: (N,)

#     # Apply mask to remove reflections that are below 1 count across all thicknesses
#     filtered_intensities = scaled_intensities[:, valid_mask]  # Shape: (M, N_filtered)
#     filtered_hkls = [tuple(h) for h in hkls_array[valid_mask]]  # Convert back to list of tuples

#     return filtered_intensities, filtered_hkls
def scale_dataset(all_intensities, all_hkls, I_max=1e8):
    """
    Scale the entire dataset so that no intensity exceeds I_max.
    Remove reflections below 1 count (across all thicknesses) on a per-pattern basis.
    """
    # 1) Determine the global maximum across all patterns
    import torch
    global_max = max(float(torch.max(I)) for I in all_intensities)
    global_scale_factor = I_max / global_max if global_max > 0 else 1.0

    scaled_data = []
    for intensities, hkls in zip(all_intensities, all_hkls):
        # Scale
        scaled_intensities = intensities * global_scale_factor

        # Filter out reflections below 1 count for this pattern
        valid_mask = (scaled_intensities >= 1).any(dim=0)
        filtered_intensities = scaled_intensities[:, valid_mask]
        hkls = np.array(hkls, dtype=int) 
        filtered_hkls = [tuple(h) for h in hkls[valid_mask]]

        scaled_data.append((filtered_intensities, filtered_hkls))

    return scaled_data

def add_poisson_noise(scaled_data, random_seed=None):
    """
    scaled_data: list of (scaled_intensities, filtered_hkls) pairs
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    noisy_data = []
    for scaled_intensities, filtered_hkls in scaled_data:
        # add Poisson noise
        noisy_intensities = torch.poisson(scaled_intensities)
        noisy_data.append((noisy_intensities, filtered_hkls))
    
    return noisy_data

def add_detector_noise(poisson_noisy_intensities, G=1.0, gamma=1.3, psi=2.0,
                       random_seed=None):
    """
    Apply a simple detector noise model to Poisson-noisy intensities.

    The final model is roughly:
      final_counts = (G * gamma) * poisson_counts + Normal(0, sqrt(psi))

    and the variance is approx:
      Var(final_counts) ~ G*gamma*poisson_counts + psi

    Parameters:
    - poisson_noisy_intensities (torch.Tensor): Intensities (already Poisson sampled).
    - G (float): Detector gain.
    - gamma (float): Detector cascade factor.
    - psi (float): Readout/electronic noise variance.
    - random_seed (int, optional): Seed for reproducibility.
    
    Adapted from
    https://journals.iucr.org/paper?S0021889810033418
    Returns:
    - final_intensities (torch.Tensor): Intensities after detector noise.
    - final_sigmas (torch.Tensor): Approx. per-pixel uncertainty (standard deviation).
    """

    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Scale the Poisson counts by (G * gamma)
    scaled_intensities = poisson_noisy_intensities * (G * gamma)

    # Add readout/electronic noise: Normal(0, sqrt(psi))
    readout_noise = torch.normal(
        mean=torch.zeros_like(scaled_intensities),
        std=torch.sqrt(torch.tensor(psi, dtype=scaled_intensities.dtype,
                                    device=scaled_intensities.device))
    )
    final_intensities = scaled_intensities + readout_noise

    # Clip negative intensities if desired
    final_intensities = torch.clamp(final_intensities, min=0.0)

    # The theoretical variance at each pixel (assuming final_intensities ~ mean):
    #     Var = G*gamma*(true mean) + psi
    # For a quick approximate "per-pixel" sigma, many users just do:
    final_sigmas = torch.sqrt(torch.clamp(G * gamma * final_intensities + psi, min=1e-8))

    return final_intensities, final_sigmas

def read_structure_factors(filename):
    data = []
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # Skip header lines
            if not parts[0].lstrip('-').isdigit():
                continue
            
            h, k, l = map(int, parts[:3])
            Re_F_paw, Im_F_paw = map(float, parts[3:5])
            Re_F_iam, Im_F_iam = map(float, parts[5:7])
            
            data.append([h, k, l, Re_F_paw, Im_F_paw, Re_F_iam, Im_F_iam])
    
    data = np.array(data)
    hkls = data[:, :3].astype(int)
    F_paw = data[:, 3] + 1j * data[:, 4]
    F_iam = data[:, 5] + 1j * data[:, 6]
    
    return hkls, F_paw, F_iam

def element_from_atomic_number(Z):
    # mapping from atomic number to element symbol
    periodic_table = {
        1:  "H",   2:  "He",  3:  "Li",  4:  "Be",  5:  "B",
        6:  "C",   7:  "N",   8:  "O",   9:  "F",   10: "Ne",
        11: "Na",  12: "Mg",  13: "Al",  14: "Si",  15: "P",
        16: "S",   17: "Cl",  18: "Ar",  19: "K",   20: "Ca",
        21: "Sc",  22: "Ti",  23: "V",   24: "Cr",  25: "Mn",
        26: "Fe",  27: "Co",  28: "Ni",  29: "Cu",  30: "Zn",
        31: "Ga",  32: "Ge",  33: "As",  34: "Se",  35: "Br",
        36: "Kr",  37: "Rb",  38: "Sr",  39: "Y",   40: "Zr",
        41: "Nb",  42: "Mo",  43: "Tc",  44: "Ru",  45: "Rh",
        46: "Pd",  47: "Ag",  48: "Cd",  49: "In",  50: "Sn",
        51: "Sb",  52: "Te",  53: "I",   54: "Xe",  55: "Cs",
        56: "Ba",  57: "La",  58: "Ce",  59: "Pr",  60: "Nd",
        61: "Pm",  62: "Sm",  63: "Eu",  64: "Gd",  65: "Tb",
        66: "Dy",  67: "Ho",  68: "Er",  69: "Tm",  70: "Yb",
        71: "Lu",  72: "Hf",  73: "Ta",  74: "W",   75: "Re",
        76: "Os",  77: "Ir",  78: "Pt",  79: "Au",  80: "Hg",
        81: "Tl",  82: "Pb",  83: "Bi",  84: "Po",  85: "At",
        86: "Rn",  87: "Fr",  88: "Ra",  89: "Ac",  90: "Th",
        91: "Pa",  92: "U",   93: "Np",  94: "Pu",  95: "Am",
        96: "Cm",  97: "Bk",  98: "Cf",  99: "Es",  100:"Fm",
        101:"Md",  102:"No",  103:"Lr",  104:"Rf",  105:"Db",
        106:"Sg",  107:"Bh",  108:"Hs",  109:"Mt",  110:"Ds",
        111:"Rg",  112:"Cn",  113:"Nh",  114:"Fl",  115:"Mc",
        116:"Lv",  117:"Ts",  118:"Og"
    }

    return periodic_table.get(Z, f"X{Z}")