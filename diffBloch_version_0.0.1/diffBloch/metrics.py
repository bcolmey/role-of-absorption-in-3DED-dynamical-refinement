import torch
from ase.build import minimize_rotation_and_translation
from ase import Atoms
import numpy as np


def get_loss(loss_name):
    """
    Get loss function from loss name
    """
    loss_dict = {
        "mse": mse_loss,
        "l1": l1_loss,
        "rbragg_abs": rbragg_abs,
        "weighted_mse": weighted_mse_loss,
        "wrbragg": wRbragg,
        "flat_bottomed_l1_loss": flat_bottomed_l1_loss,
        "l2": l2_loss,
        "flat_bottomed_percentage_difference": flat_bottomed_percentage_difference
    }
    if loss_name not in loss_dict:
        raise ValueError(f"Loss name {loss_name} not found")
    return loss_dict[loss_name]


def euclidean_distance(optimised_coords, true_positions):
    """
    Compute euclidean distance between optimised coordinates and true positions
    Args:
        optimised_coords: torch.tensor of shape (N, 3)
        true_positions: torch.tensor of shape (N, 3)
    Returns:
        dist: torch.tensor of shape (N,)
        avg_dist: torch.tensor of shape (1,)
    """
    diff = optimised_coords - true_positions
    dist_squared = torch.sum(diff * diff, dim=1)
    dist = torch.sqrt(dist_squared)
    avg_dist = dist.mean()
    return dist, avg_dist


def rmsd(
    reference_positions,
    model,
    hydrogen=False,
    align=True,
):
    """
    Computes the root-mean-square deviation (RMSD) between the atomic positions 
    of a reference structure and a model structure.

    Parameters:
    ----------
    reference_positions : torch.Tensor
        The fractional atomic positions of the reference structure (shape: [N, 3]).
    model : object
        A model containing atomic information, including atomic numbers, positions, 
        and unit cell parameters.
    hydrogen : bool, optional (default=False)
        If False, hydrogen atoms are excluded from the RMSD calculation.
    align : bool, optional (default=True)
        If True, the model structure is aligned to the reference structure 
        by minimizing rotational and translational differences.

    Returns:
    -------
    float
        The RMSD value between the reference and model atomic positions.
    
    Notes:
    -----
    - Assumes the atomic order in `reference_positions` and `model.atoms` is the same.
    - Converts fractional coordinates to Cartesian using the unit cell.
    - If `align` is True, the structures are superimposed before calculating RMSD.
    - Excludes hydrogen atoms if `hydrogen=False`.
    """
    #assumes atoms are in same order in ref and model, might break if not careful
    atomic_numbers_ref = model.atoms.asu_numbers.detach().cpu().numpy()
    atomic_numbers_model = model.atoms.asu_numbers.detach().cpu().numpy()
    model_positions = model.atoms.asu_positions.detach().cpu().numpy()
    reference_positions = reference_positions.detach().cpu().numpy()
    #convert to cartesian posi tions
    unit_cell = model.atoms.unit_cell
    reference_positions = (unit_cell.T @ reference_positions.T).T
    model_positions = (unit_cell.T @ model_positions.T).T

    reference_structure = Atoms(
        numbers=atomic_numbers_ref, positions=reference_positions
    )
    model_structure = Atoms(
        numbers=atomic_numbers_model, positions=model_positions
    )
    if not hydrogen:
        reference_structure = reference_structure[
            [atom.index for atom in reference_structure if atom.symbol != "H"]
        ]
        model_structure = model_structure[
            [atom.index for atom in model_structure if atom.symbol != "H"]
        ]
    if align:
        minimize_rotation_and_translation(reference_structure, model_structure)
    rmsd = np.sqrt(
        ((reference_structure.positions - model_structure.positions) ** 2).sum()
        / len(model_structure.positions)
    )
    return rmsd


def l1_loss(sim_intensities, exp_intensities, sigmas=None):
    """
    Compute L1 loss (without weights)
    """
    return torch.mean(torch.abs(sim_intensities - exp_intensities), dim=-1)


def mse_loss(sim_intensities, exp_intensities, sigmas=None):
    """
    Compute MSE loss
    """
    return torch.mean((sim_intensities - exp_intensities) ** 2, dim=-1)


def weighted_mse_loss(sim_intensities, exp_intensities, sigmas):
    """
    Compute weighted MSE loss
    Params:
    - pred: torch.tensor of predicted values
    - target: torch.tensor of target values
    - sigma: torch.tensor of weights
    Returns:
    - loss: torch.tensor of loss
    """
    weight = 1 / sigmas**2
    return torch.sum(weight * (sim_intensities - exp_intensities) ** 2, dim=-1)

def l2_loss(pred, target):
    """
    Compute the L2 loss (squared error) between the predicted and target values.
    
    Params:
    - pred: torch.tensor of predicted values
    - target: torch.tensor of target values
    
    Returns:
    - loss: torch.tensor, the computed L2 loss
    """
    # Calculate the squared difference
    squared_diff = (pred - target) ** 2
    # Return the mean loss
    return torch.mean(squared_diff)

def flat_bottomed_percentage_difference(pred, target, tau):
    """

    """
    epsilon = 1e-6  # Small value to avoid division by zero

    percentage_difference = torch.abs(target - pred) / torch.abs(target + epsilon) * 100
    # non_zero_mask = percentage_difference != 0
    # non_zero_values = percentage_difference[non_zero_mask]

    # # Only print non-zero percentage changes
    # if len(non_zero_values) > 0:
    #     print(f'percentage_change: {non_zero_values}')
    loss = torch.clamp(percentage_difference - tau, min=0)
    # Return the mean loss
    return torch.mean(loss)

def flat_bottomed_l1_loss(pred, target, tau):
    """
    Adapted from Jumper et al.,https://doi.org/10.1038/s41586-021-03819-2

    Compute flat-bottomed L1 loss for bond distances. The loss is zero if the absolute
    difference between prediction and target is within a margin defined by tau, which
    is a multiple of the standard deviation from literature bond angles.

    Params:
    - pred: torch.tensor of predicted values
    - target: torch.tensor of target values
    - tau: float, the tolerance margin within which no loss is computed

    Returns:
    - loss: torch.tensor, the computed loss
    """

    # Ensure tau is a tensor, if it's a float convert it
    if isinstance(tau, float):
        tau = torch.tensor(tau, device=pred.device)

    # Calculate the absolute difference
    abs_diff = torch.abs(pred - target)
    # Subtract tau from the absolute differences and clamp at zero
    # to create the flat bottom
    loss = torch.clamp(abs_diff - tau, min=0)
    # Return the mean loss
    return torch.mean(loss)


def rbragg_abs(sim_intensities, exp_intensities, sigmas=None):
    """
    Compute the Rbragg metric using only reflections where I_exp > 3 * sigma.

    R = sum(|sqrt(I_pred) - sqrt(I_exp)|) / sum(sqrt(I_exp))

    Params:
    - sim_intensities: torch.tensor of predicted intensities
    - exp_intensities: torch.tensor of experimental intensities
    - sigmas: torch.tensor of experimental uncertainties (same shape as intensities)

    Returns:
    - R(obs): torch.tensor of R(obs) value
    """
    if sigmas is None:
        raise ValueError("sigmas must be provided to apply I > 3*sigma filter.")

    mask = exp_intensities > 3 * sigmas
    sqrt_exp = exp_intensities.sqrt()
    sqrt_sim = sim_intensities.sqrt()
    num = torch.sum(torch.abs(sqrt_exp - sqrt_sim) * mask, dim=-1)
    denom = torch.sum(sqrt_exp * mask, dim=-1)
    return num / denom

def wRbragg(sim_intensities, exp_intensities, sigmas, mu=0.01):
    """
    Compute the weighted Rbragg metric:
    wR2 = sqrt( sum(w*(I_pred - I_exp)^2) / sum(w*I_exp^2) )
    where:
        w = 1 / sqrt( σ(sqrt(I_exp))^2 + (μ * sqrt(I_exp))^2 )

    Following SI of https://doi.org/10.1038/s41557-023-01186-1

    Parameters:
    - sim_intensities: torch.tensor of predicted intensities (I_calc)
    - exp_intensities: torch.tensor of experimental intensities (I_obs)
    - sigmas: torch.tensor of σ(I_obs)
    - mu: float, instability factor (default=0.01)

    Returns:
    - rbragg: torch.tensor of wR value
    """

    eps = 1e-12
    sqrt_I = torch.sqrt(torch.clamp(exp_intensities, min=eps))

    # Define mask for weak reflections: I < 0.01 * sigma
    weak_mask = exp_intensities < (0.01 * sigmas)

    # Compute σ(sqrt(I))
    sigma_sqrt_I = torch.empty_like(sigmas)
    sigma_sqrt_I[weak_mask] = 5 * torch.sqrt(sigmas[weak_mask])
    sigma_sqrt_I[~weak_mask] = 0.5 * sigmas[~weak_mask] / sqrt_I[~weak_mask]

    # Final weights
    w = 1.0 / torch.sqrt(sigma_sqrt_I**2 + (mu * sqrt_I)**2)

    # Compute wR2
    numerator = torch.sum((w * (sim_intensities - exp_intensities)) ** 2, dim=-1)
    denominator = torch.sum((w * exp_intensities) ** 2, dim=-1)

    return torch.sqrt(numerator / denominator)

def convexity_loss_fn(alpha_min, alpha_max, bloch_nn):
    """
    Compute the convexity loss for the thickness neural network
    Params:
    - alpha_min: float, minimum angle for evaluation
    - alpha_max: float, maximum angle for evaluation
    - bloch_nn: BlochNN object
    Returns:
    - loss: torch.tensor, the computed loss
    - predictions: list of predictions over theta values for logging
    """
    data = []
    mu_predictions = []
    # create theta values to evaluate convexity loss over
    #NN always trained on normalised alphas betwen -1 and 1
    x = torch.linspace(-1, 1, 200, dtype = torch.float64, requires_grad=True)
    plotting_data = torch.linspace(alpha_min, alpha_max, 200, dtype = torch.float64)
    # accumulate predictions from nn over theta values
    for i, angle in enumerate(x):
        mu_pred = bloch_nn.thickness_nn(angle.to(bloch_nn.device))[0]
        mu_predictions.append(mu_pred)
        data.append([plotting_data[i].item(), mu_pred.item()])
    y = torch.stack(mu_predictions)
    dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    d2y_dx2 = torch.autograd.grad(dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=True)[0]
    return torch.mean(torch.clamp(-d2y_dx2, min = 0) ** 2), data
