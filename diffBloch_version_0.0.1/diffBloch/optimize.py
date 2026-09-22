"""
Optimization routines for the refinement of the structures.
"""

import torch

from diffBloch.metrics import get_loss


class BlochLoss(torch.nn.Module):
    """
    Loss function for the Bloch wave simulation.
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        self.diffraction_loss_fn = get_loss(cfg.loss.diffraction_loss)

        #### Regularization losses
        # bond length loss
        self.bond_length_loss = cfg.loss.bond_length_loss
        self.bond_length_loss_weight = 0.0
        self.max_bond_length_loss_weight = cfg.loss.bond_length_loss_weight
        # angle loss
        self.angle_loss = cfg.loss.angle_loss
        self.angle_loss_weight = 0.0
        self.max_angle_loss_weight = cfg.loss.angle_loss_weight
        # plane loss
        self.plane_loss = cfg.loss.plane_loss
        self.plane_loss_weight = 0.0
        self.max_plane_loss_weight = cfg.loss.plane_loss_weight
        # multiplier of sigma for flat bottomed loss etc.
        self.restraint_sigma = cfg.loss.restraint_sigma * 2
        # rigid bond loss
        self.rigid_bond_loss = cfg.loss.rigid_bond_loss
        self.rigid_bond_loss_weight = 0.0
        self.max_rigid_bond_loss_weight = cfg.loss.rigid_bond_loss_weight
        # Vg symmetry loss
        self.vg_symmetry_mag_loss = cfg.loss.vg_symmetry_mag_loss
        self.vg_symmetry_phase_loss = cfg.loss.vg_symmetry_phase_loss
        self.vg_symmetry_mag_loss_weight = 0.0
        self.vg_symmetry_phase_loss_weight = 0.0
        self.max_vg_symmetry_mag_loss_weight = cfg.loss.vg_symmetry_mag_loss_weight
        self.max_vg_symmetry_phase_loss_weight = cfg.loss.vg_symmetry_phase_loss_weight
        # VG magnitude loss
        self.vg_percentage_change_loss = get_loss(cfg.loss.vg_percentage_change_loss)
        self.vg_percentage_change_loss_weight = cfg.loss.vg_percentage_change_loss_weight
        # annealing schedule
        self.constraint_anneal_schedule = cfg.loss.constraint_anneal_schedule
        self.constraint_anneal_steps = cfg.loss.constraint_anneal_steps

    def anneal_losses(self, step):
        """
        Anneal the losses.
        Start from no constraint to max weight in init.
        """
        if self.constraint_anneal_schedule == "linear":
            if step < self.constraint_anneal_steps:                
                self.bond_length_loss_weight = (
                    self.max_bond_length_loss_weight * step / self.constraint_anneal_steps
                )
                self.angle_loss_weight = (
                    self.max_angle_loss_weight * step / self.constraint_anneal_steps
                )
                self.plane_loss_weight = (
                    self.max_plane_loss_weight * step / self.constraint_anneal_steps
                )
                self.rigid_bond_loss_weight = (
                    self.max_rigid_bond_loss_weight * step / self.constraint_anneal_steps
                )
                self.vg_symmetry_mag_loss_weight = (
                    self.max_vg_symmetry_mag_loss_weight * step / self.constraint_anneal_steps
                )
                self.vg_symmetry_phase_loss_weight = (
                    self.max_vg_symmetry_phase_loss_weight * step / self.constraint_anneal_steps
                )
            else:
                self.bond_length_loss_weight = self.max_bond_length_loss_weight
                self.angle_loss_weight = self.max_angle_loss_weight
                self.plane_loss_weight = self.max_plane_loss_weight
                self.rigid_bond_loss_weight = self.max_rigid_bond_loss_weight
                self.vg_symmetry_mag_loss_weight = self.max_vg_symmetry_mag_loss_weight
                self.vg_symmetry_phase_loss_weight = self.max_vg_symmetry_phase_loss_weight
        elif self.constraint_anneal_schedule == "constant":
            self.bond_length_loss_weight = self.max_bond_length_loss_weight
            self.angle_loss_weight = self.max_angle_loss_weight
            self.plane_loss_weight = self.max_plane_loss_weight
            self.rigid_bond_loss_weight = self.max_rigid_bond_loss_weight
            self.vg_symmetry_mag_loss_weight = self.max_vg_symmetry_mag_loss_weight
            self.vg_symmetry_phase_loss_weight = self.max_vg_symmetry_phase_loss_weight
        else:
            raise ValueError(
                f"Constraint annealing schedule {self.constraint_anneal_schedule} not supported"
            )

    def forward(self, simulated_intensities, exp_intensities, exp_sigmas, bloch_nn=None):
        """
        Compute the loss function for the Bloch wave simulation
        Params:
        - simulated_intensities: torch.tensor of simulated intensities
        - exp_intensities: torch.tensor of experimental intensities
        - exp_sigmas: torch.tensor of experimental sigmas
        - bloch_nn: bloch module containing atoms_nn with asu, bond pairs, thermal displacements
        Returns:
        - loss: dict of torch.tensors of different losses
        """
        diffraction_loss = self.diffraction_loss_fn(
            simulated_intensities, exp_intensities, exp_sigmas
        )
        if bloch_nn is None or self.rigid_bond_loss_weight == 0.0:
            # print("No bloch_nn provided or self.rigid_bond_loss_weight==0.0, skipping rigid bond loss, expected for Vg refinement")
            rigid_bond_loss = torch.tensor(0.0)
        else:
            rigid_bond_loss = rigid_bond_loss_fn(bloch_nn.structure_factor_net.atoms, criterion=self.rigid_bond_loss)

        if bloch_nn is None or self.bond_length_loss_weight == 0.0:
            bond_length_loss = torch.tensor(0.0)
        else:
            bond_length_loss = bond_length_loss_fn(bloch_nn.structure_factor_net.atoms, criterion=self.bond_length_loss)
        
        if bloch_nn is None or self.angle_loss_weight == 0.0:
            angle_loss = torch.tensor(0.0)
        else:
            angle_loss = angle_loss_fn(bloch_nn.structure_factor_net.atoms, criterion=self.angle_loss)
        
        if bloch_nn is None or self.plane_loss_weight == 0.0:
            plane_loss = torch.tensor(0.0)
        else:
            plane_loss = plane_loss_fn(bloch_nn.structure_factor_net.atoms, criterion=self.plane_loss)

        if bloch_nn is None or self.vg_symmetry_mag_loss_weight == 0.0:
            # print("No bloch_nn provided or self.vg_symmetry_mag_loss_weight==0.0, skipping Vg symmetry loss")
            vg_sym_mag_loss = torch.tensor(0.0)
            vg_sym_phase_loss = torch.tensor(0.0)
            vg_percentage_change_loss = torch.tensor(0.0)
        elif not bloch_nn.refine_vg:
            vg_sym_mag_loss = torch.tensor(0.0)
            vg_sym_phase_loss = torch.tensor(0.0)
            vg_percentage_change_loss = torch.tensor(0.0)
        else:
            vg_sym_mag_loss, vg_sym_phase_loss = symmetry_in_vg_loss_fn(bloch_nn, mag_criterion=self.vg_symmetry_mag_loss, phase_criterion=self.vg_symmetry_phase_loss)
            vg_percentage_change_loss = self.vg_percentage_change_loss(bloch_nn.initial_Vg, bloch_nn.Vg, tau = 5)

        total_loss = diffraction_loss + self.rigid_bond_loss_weight * rigid_bond_loss + self.vg_symmetry_mag_loss_weight * vg_sym_mag_loss + self.vg_symmetry_phase_loss_weight * vg_sym_phase_loss + self.vg_percentage_change_loss_weight * vg_percentage_change_loss + self.bond_length_loss_weight * bond_length_loss + self.angle_loss_weight * angle_loss + self.plane_loss_weight * plane_loss

        losses = { 
            "diffraction_loss": diffraction_loss,
            "bond_length_loss": bond_length_loss,
            "angle_loss": angle_loss,
            "plane_loss": plane_loss,
            "rigid_bond_loss": rigid_bond_loss,
            "vg_symmetry_mag_loss": vg_sym_mag_loss,
            "vg_symmetry_phase_loss": vg_sym_phase_loss,
            "vg_percentage_change_loss": vg_percentage_change_loss,
            "total_loss": total_loss,
            "bond_length_loss_weight": self.bond_length_loss_weight,
            "angle_loss_weight": self.angle_loss_weight,
            "plane_loss_weight": self.plane_loss_weight,
            "rigid_bond_loss_weight": self.rigid_bond_loss_weight,
            "vg_symmetry_mag_loss_weight": self.vg_symmetry_mag_loss_weight,
            "vg_symmetry_phase_loss_weight": self.vg_symmetry_phase_loss_weight,
            "vg_percentage_change_loss_weight": self.vg_percentage_change_loss_weight,
        }

        return losses


    def __repr__(self):
        return f"BlochLoss(diffraction_loss={self.diffraction_loss_fn}, rigid_bond_loss={self.rigid_bond_loss}, vg_symmetry_mag_loss={self.vg_symmetry_mag_loss}, vg_symmetry_phase_loss={self.vg_symmetry_phase_loss}, vg_percentage_change_loss={self.vg_percentage_change_loss}, bond_length_loss={self.bond_length_loss_fn}, angle_loss={self.angle_loss_fn}, plane_loss={self.plane_loss_fn})"


def bond_length_loss_fn(atoms_nn, criterion='mse'):
    if atoms_nn.bond_constraints is None:
        return torch.tensor(0.0)
    asu_cart = atoms_nn.get_asu_cartesian_coords()
    bonds = atoms_nn.bond_constraints_idxs
    bond_vectors = asu_cart[bonds[:,1]] - asu_cart[bonds[:,0]]    # (N_bonds, 3)
    bond_lengths = torch.norm(bond_vectors, dim=1)  # (N_bonds)
    bond_loss = get_loss(criterion)(bond_lengths, atoms_nn.bond_constraints[:,0], atoms_nn.bond_constraints[:,1])
    return bond_loss


def angle_loss_fn(atoms_nn, criterion='mse'):
    if atoms_nn.angle_constraints is None:
        return torch.tensor(0.0)
    asu_cart = atoms_nn.get_asu_cartesian_coords()  # (N_atoms, 3)
    bonds = atoms_nn.angle_constraints_idxs # (N_angles, 3)
    vec1 = asu_cart[bonds[:, 1]] - asu_cart[bonds[:, 0]]    # (N_angles, 3)
    vec2 = asu_cart[bonds[:, 1]] - asu_cart[bonds[:, 2]]    # (N_angles, 3)
    cos_angles = (vec1*vec2).sum(dim=1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1))    # (N_angles)
    current_angles = torch.acos(cos_angles.clamp(-1 + 1e-7, 1 - 1e-7)) * (180 / torch.pi)  # Convert to degrees
    angle_loss = get_loss(criterion)(current_angles, atoms_nn.angle_constraints[:, 0], atoms_nn.angle_constraints[:, 1]*4)
    
    return angle_loss


def plane_loss_fn(atoms_nn, criterion='mse'):
    # iterate through planes
    asu_cart = atoms_nn.get_asu_cartesian_coords()  # (N_atoms, 3)
    planes = atoms_nn.plane_constraints_idxs
    plane_loss = torch.tensor(0.0, device=asu_cart.device)
    for plane_idxs in planes:
        # Extract the positions of the atoms that should lie on the plane
        plane_atoms =  asu_cart[plane_idxs]  # Shape (M, 3) where M is the number of atoms for the plane

        # Calculate the centroid of these atoms
        centroid = plane_atoms.mean(dim=0, keepdim=True)  # Shape (1, 3)

        # Subtract the centroid to center the atoms at the origin
        centered_positions = plane_atoms - centroid

        # Calculate the covariance matrix of the centered positions
        covariance_matrix = centered_positions.T @ centered_positions  # Shape (3, 3)

        # Find the normal vector of the plane using the smallest eigenvector of the covariance matrix
        eigvals, eigvecs = torch.linalg.eigh(covariance_matrix)
        plane_normal = eigvecs[:, 0]  # Eigenvector corresponding to the smallest eigenvalue

        # Calculate the distance of each atom from the plane
        distances = (centered_positions @ plane_normal).abs()  # Shape (M,)

        sigma = torch.zeros_like(distances) + 0.02        
        plane_loss += get_loss(criterion)(distances, torch.zeros_like(distances), sigma)
        
    return plane_loss


def symmetry_in_vg_loss_fn(bloch_nn, mag_criterion="flat_bottomed_l1_loss", phase_criterion="mse"):
    """
    Computes the symmetry constraint loss, ensuring Vgs related by symmetry are equal.
    Parameters:
    - bloch_nn: model containing the Vgs
    - criterion: The criterion to use for the loss. Default is L2 loss.
    Returns:
    - symmetry_loss: The total L2 loss term corresponding to the symmetry constraint, scalar.
    """
    # mag loss - make magnitude of symmetry related Vgs equal
    # use 1e-2 tolerance for flat_bottomed_l1 loss on mag
    mags = bloch_nn.Vg.abs()
    mag_loss = get_loss(mag_criterion)(mags[bloch_nn.Vg_symmetry_constraints_idxs[:,0]], mags[bloch_nn.Vg_symmetry_constraints_idxs[:,1]], 1e-2)
    
    # phase loss - make phase of symmetry related Vgs equal to the original Vg
    phases = bloch_nn.Vg.angle()
    phase_diff = 1-torch.cos(phases[bloch_nn.Vg_symmetry_constraints_idxs[:,1]] - phases[bloch_nn.Vg_symmetry_constraints_idxs[:,0]])
    phase_loss = get_loss(phase_criterion)(phase_diff, bloch_nn.Vg_phase_diff)
    return mag_loss, phase_loss


def rigid_bond_loss_fn(atoms_nn, criterion='mse'):
    """
    Computes the rigid bond constraint loss.
    
    Parameters:
    - atoms_nn: Atoms object with the atomic positions, bonds and anisotropic displacement parameters.
    - criterion: The criterion to use for the loss. Default is L2 loss.    
    Returns:
    - rigid_bond_loss: The total L2 loss term corresponding to the rigid bond constraint, scalar.
    """

    # use bond pairs to get bond vectors in the ASU
    asu = atoms_nn.asu_positions
    bonds = atoms_nn.bond_pairs
    bond_vectors = asu[bonds[:,1]] - asu[bonds[:,0]]    # (N_bonds, 3)
    bond_directions = bond_vectors / torch.norm(bond_vectors, dim=1, keepdim=True)  # Normalize bond vectors
    
    # Get the anisotropic displacement parameters for the atoms in the bond pairs
    Uij_atoms = atoms_nn.Uij_layer.get_all_Uij_tensor()  # (N_atoms, 3, 3)    # TODO maybe save this instead of computing every time?
    Uij_bond_1 = Uij_atoms[bonds[:,0]]  # (N_bonds, 3, 3)
    Uij_bond_2 = Uij_atoms[bonds[:,1]]  # (N_bonds, 3, 3)

    # Project Uani_1 and Uani_2 onto bond directions (N_bonds)
    U1 = torch.einsum('bi,bij,bj->b', bond_directions, Uij_bond_1, bond_directions)
    U2 = torch.einsum('bi,bij,bj->b', bond_directions, Uij_bond_2, bond_directions)

    rigid_bond_loss = get_loss(criterion)(U1, U2)

    return rigid_bond_loss


def init_optim(cfg, model):
    """
    Initialize optimizer, scheduler, and criterion for multiple models.

    Args:
        cfg: Configuration object with optimizer and scheduler settings.
        model: PyTorch model to optimize.

    Returns:
        criterion: PyTorch loss function.
        optimizer: PyTorch optimizer for both models.
        scheduler: PyTorch learning rate scheduler.
    """
    
    # Ensure all model parameters are contiguous
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            print(f"Parameter '{name}' in {model.__class__.__name__} is not contiguous. Making it contiguous.")
            param.data = param.data.contiguous()

    # Loss function
    criterion = BlochLoss(cfg)

    # Collect parameter groups from both models
    params = [
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if "thermal_displacements" not in name and "thickness_nn" not in name
            ],
            "lr": cfg.optimizer.lr,
        },
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if "thermal_displacements" in name
            ],
            "lr": cfg.optimizer.lr_disps,
        },
        {
            "params": [
                p for name, p in model.thickness_nn.named_parameters()
            ],
            "lr": cfg.optimizer.lr_thickness_model,
        }
    ]

    # different learning rate for thermal_displacements
    if cfg.optimizer.name == "lbfgs":
        params = model.parameters()
        optimizer = torch.optim.LBFGS(
            params,
            lr=cfg.optimizer.lr,
            max_iter=5,
            history_size=5,
            line_search_fn="strong_wolfe",
        )
        scheduler = None
    else:
        if cfg.optimizer.name == "adam":
            optimizer = torch.optim.Adam(
                params,
                weight_decay=cfg.optimizer.weight_decay,
            )
        elif cfg.optimizer.name == "sgd":
            optimizer = torch.optim.SGD(
                params,
                weight_decay=cfg.optimizer.weight_decay,
            )
        elif cfg.optimizer.name == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                weight_decay=cfg.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {cfg.optimizer.name} not supported")

    if cfg.scheduler.name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma,
        )
    elif cfg.scheduler.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, verbose=True
        )
    elif cfg.scheduler.name == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=cfg.epochs,
        )
    elif cfg.scheduler.name == "cyclical":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=cfg.optimizer.lr,
            max_lr=cfg.optimizer.lr,
            step_size_up=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma,
            mode="exp_range",
            cycle_momentum=False,
        )
    elif cfg.scheduler.name == "constant" or cfg.scheduler.name is None:
        scheduler = None
    else:
        raise ValueError(f"Scheduler {cfg.scheduler.name} not supported")
    
    
    return criterion, optimizer, scheduler
