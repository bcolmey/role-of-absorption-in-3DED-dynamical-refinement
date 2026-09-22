import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
torch.set_num_threads(5)
torch.set_num_interop_threads(4)
#print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
#print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")
#print(f"PyTorch intra-op threads: {torch.get_num_threads()}")
#print(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")
#import time

import numpy as np

from typing import Sequence
from scipy.constants import c, h, e, m_e
from abtem.core.energy import energy2wavelength, energy2sigma
from abtem.core.constants import kappa
from abtem.parametrizations import Parametrization, validate_parametrization
from diffBloch.utils import (
    excitation_errors,
    reciprocal_cell,
    get_reflection_condition,
    make_hkl_grid,
    reciprocal_space_gpts,
)
from diffBloch.complex_scattering_factor import calc_complex_scattering_factors
from diffBloch.atoms import Atoms
from diffBloch.diffraction_dataset import DiffractionDataset
from diffBloch.rotation_dataset import rotation_matrix_z
from diffBloch.utils import (
    calculate_g_vec, reciprocal_cell, 
    raveled_hkl_to_hkl_torch, 
    fill_diagonal_torch,
    filter_reciprocal_space_vectors,
    gmax_mask,
    create_Fgb_symmetry_restraints,
    sample_thicknesses,
)

# Headroom (1/Å) added to the structure-factor table beyond twice the beam cutoff. It only enlarges the
# unused edge of the table, so it changes neither the beams in the Bloch solve nor the scored reflections.
SUPPORT_MARGIN = 0.5


class StructureFactorNet(nn.Module):
    """
    The StructureFactorNet class calculates the structure factors for a given set of
    atoms and parametrization. This codebase is based on the abtem bloch wave library
    but has been modified to be differentiable wrt atomic positions in the asymmetri unit.

    Parameters
    ----------
    atoms_nn : Atoms
        initialised Atoms neuralnet object.
    solve_g_max : float
        Beam cutoff [1/Å]: the largest |g| of the beams coupled in one Bloch solve (``bloch.g_max``). The
        structure-factor table is derived from it and extends to ``2 * solve_g_max + SUPPORT_MARGIN``,
        because the structure matrix needs F(g_j - g_i) and those differences reach ``2 * solve_g_max``.
        The structure-factor table radius is therefore never entered separately.
    parameterization : str, optional
        Parameterization for the scattering factors. Default is 'lobato'.
    thermal_disps : float or dict, optional
        Standard deviation of the atomic displacements for the Debye-Waller factor [Å].
        Default is 0.01.
    occupancy : float, optional
        The occupancy of the atoms. Default is 1.0.
    cutoff : {'taper', 'hard'}, optional
        Cutoff function for the scattering factors. 'taper' is a smooth cutoff, 'hard'
        is a hard cutoff. Default is 'taper'.
    device : {'cpu', 'gpu'}, optional
        Device to use for calculations. Can be 'cpu' or 'gpu'. Default is 'cpu'.
    centering : {'P', 'I', 'A', 'B', 'C', 'F'}, optional
        Lattice centering. Default is 'P'.
    """
    def __init__(
        self,
        cfg,
        atoms_nn,
        thickness_nn,
        solve_g_max,
    ):
        

        super(StructureFactorNet, self).__init__()
        self.cfg = cfg
        self.atoms = atoms_nn
        if "g_max" in cfg:
            raise ValueError(
                "structure_factor.g_max is no longer used: set the beam cutoff bloch.g_max and pass it as "
                "solve_g_max. The structure-factor table radius is derived as 2 * g_max + SUPPORT_MARGIN."
            )
        if solve_g_max <= 0:
            raise ValueError("solve_g_max must be positive")
        self.solve_g_max = float(solve_g_max)
        self.g_max = 2.0 * self.solve_g_max + SUPPORT_MARGIN  # structure-factor table radius
        self.parameterization = cfg.parameterization
        self.isotropic = self.atoms.isotropic        
        self.absorption = cfg.absorption
        self.absorption_type = cfg.absorption_type
        self.absorption_percent = cfg.absorption_percent
        self.energy = cfg.energy
        self.U_0_prime = None
        self.thickness_nn = thickness_nn
        self.velocity = None

        
        if self.absorption_type == "param":
            gamma = 1 + (e*self.energy)/(m_e*c**2)
            velocity = c*np.sqrt(1 - 1/gamma**2)
            self.velocity = velocity


        if cfg.cutoff not in ("taper", "hard"):
            raise ValueError("cutoff must be 'taper', 'hard'")


        self.cutoff = cfg.cutoff
        self.device = cfg.device
        
        self.centering = atoms_nn.centering
        if not atoms_nn.centering:
            warnings.warn("Centering not specified, assuming primitive (P).")

        hkl = make_hkl_grid(self.atoms.unit_cell, self.g_max)
        if self.atoms.centering.lower() != "p":
            hkl = hkl[get_reflection_condition(hkl, self.atoms.centering)]

        self.hkl = torch.tensor(hkl, dtype=self.atoms.asu_positions.dtype, device = self.device)
        self.gpts = reciprocal_space_gpts(self.atoms.unit_cell, self.g_max)
        # convert to ints
        self.gpts = tuple(map(int, self.gpts))
        self.gvec = hkl @ self.atoms.reciprocal_cell()

        self.g_vec_length = np.linalg.norm(self.gvec, axis=1)


    def forward(self):
        # TODO, might be best to prebuild scattering factors, although doesn't seem as problematic as the 2d multislice case
        
        # the below line expands the asu
        expanded_positions, expanded_atomic_numbers, expanded_disps, occupancy = self.atoms()

        if self.cfg.overwrite_occupancy:
            occupancy = np.ones_like(occupancy) * self.cfg.occupancy
        absorption = self.absorption
        absorption_type = self.absorption_type
        absorption_percent = self.absorption_percent
        velocity = self.velocity
        fe = self.calculate_scattering_factors(
            g=self.g_vec_length,
            positions=expanded_positions,
            numbers=expanded_atomic_numbers,
            velocity = velocity,
            cell_volume=self.atoms.cell_volume(),
            parametrization=self.parameterization,
            g_max=self.g_max,
            thermal_disps=expanded_disps,
            occupancy=occupancy, 
            hkls=self.hkl,
            reciprocal_cell=self.atoms.reciprocal_cell(),
            cutoff=self.cutoff,
            isotropic = self.isotropic,
            absorption = self.absorption,
            absorption_type = self.absorption_type,
        )
        self.fe = fe    # (N_atoms, N_hkls)

        #abtem seems to calculate electron structure factors with within the Born approximation context
        #Fgb below is related to Fgb by (Fgb = 47.878009/cell_volume * Fgb) as per Spence and Zup in their Electron microdiffraction book (page 33). 
        #In abtem, the constant 47.878009 is 1/kappa and is applied when converting structure factors below to Ug in the structure matrix equation
        
        Fgb_unmasked = torch.sum(
            fe * torch.exp(2.0j * torch.pi * (expanded_positions @ self.hkl.T)),
            dim=0
        ) 

        #numerical precision
        threshold = 1e-12
        mask_real = (torch.abs(Fgb_unmasked.real) >= threshold)
        mask_imag = (torch.abs(Fgb_unmasked.imag) >= threshold)

        new_real = torch.where(mask_real, Fgb_unmasked.real, torch.zeros_like(Fgb_unmasked.real))
        new_imag = torch.where(mask_imag, Fgb_unmasked.imag, torch.zeros_like(Fgb_unmasked.imag))

        Fgb = torch.complex(new_real, new_imag)


        if absorption:
            if absorption_type == "param":
                    if (self.U_0_prime is None):
                        """
                        Compute the absorptive component of the scattering potential (U0'). Function updates self.U_0_prime
                        which is called in bloch net method to create the structure matrix.
                        """
                        print("U_0_prime not found in StructureFactorNet. Computing it now...")
                        # Compute scattering factors for s = 0
                        prefactor = energy2sigma(self.energy) / (kappa * energy2wavelength(self.energy) * np.pi)
                        scattering_0 = torch.tensor([
                            calc_complex_scattering_factors([0], 8 * (np.pi**2) * (expanded_disps[i].trace() / 3).item(), velocity,
                                        expanded_atomic_numbers[i].item())
                            for i in range(len(expanded_positions))
                        ])
                        # Compute total f'0 and normalize by cell volume
                        total_f0_prime = torch.sum(scattering_0)
                        self.U_0_prime = prefactor*(total_f0_prime / self.atoms.cell_volume()) #prefactor converts this from F_0_born/volume to U_0)
                        print(f"Computed U_0_prime: {self.U_0_prime}")


            if (absorption_type == "constant"): 
                #("Structure factors  before imaginary:", struct_factors[:10])  # Sample of struct_factors
                Fgb_real = Fgb  # Real part of the structure factor
                Fgb_imag =  (absorption_percent / 100) * Fgb_real  # Imaginary part 
                Fgb = Fgb_real + 1j * (Fgb_imag) # Combine real and imaginary components
            
        structure_factors = Fgb / self.atoms.cell_volume()

        return structure_factors
        
    def calculate_dwf_factor(
        self,
        thermal_disps: torch.tensor,
        hkls: np.ndarray,
        reciprocal_cell: np.ndarray,
    ):
        """Calculate the Debye-Waller factor for a given set of atoms.

        Parameters
        ----------
        thermal_disps : torch.tensor
            Nx3x3 tensor of thermal displacements for each atom in the unit cell.
        hkls : np.ndarray
            The reciprocal space vectors as Miller indices. Given as a (N, 3) array.
        reciprocal_cell : np.ndarray
            The reciprocal cell.

        Returns
        -------
        torch.tensor
            The Debye-Waller factor.
        """
        reciprocal_cell = torch.tensor(reciprocal_cell, device=thermal_disps.device)
        orthog_matrix_inv = torch.tensor(self.atoms.inverse_orthogonalization_matrix(), dtype = torch.float64, device  = self.device)
        Ustar = torch.einsum("xy,ayz,wz->axw", orthog_matrix_inv, thermal_disps, orthog_matrix_inv)
        #Ustar = torch.einsum("xy,ayz,wz->axw", reciprocal_cell, thermal_disps, reciprocal_cell)
        DWF = torch.exp(-2.0 * np.pi**2 * torch.einsum("rx,axy,ry->ar", hkls, Ustar, hkls))
        return DWF
    
    def calculate_scattering_factors(
        self,
        g: np.ndarray,
        positions: torch.tensor,
        numbers: torch.tensor,
        velocity: float,
        parametrization: str | Parametrization,
        g_max: float,
        thermal_disps: torch.tensor,
        occupancy: float,
        cell_volume: float,
        hkls: np.ndarray,
        reciprocal_cell: np.ndarray,
        cutoff: str = "taper",
        isotropic: bool = True,
        absorption: bool = False,
        absorption_type: str = "param", 
    ):
        """Calculate the scattering factors for a given set of atoms and parametrization.

        Parameters
        ----------
        g : np.ndarray
            The scattering vector lengths [1/Å].
        positions : torch.tensor
            tensor of expanding atomic positions
        g_max : float
            Maximum scattering vector length [1/Å]. The scattering factors are set to zero
            for g > g_max.
        parametrization : {'lobato', 'kirkland', 'peng'}
            Parametrization for the scattering factors.
        thermal_disps : torch.tensor
            Nx3x3 tensor of thermal displacements for each atom in the unit cell.
        cutoff : {'taper', 'hard'}
            Cutoff function for the scattering factors. 'taper' is a smooth cutoff, 'hard'
            is a hard cutoff.
        hkls : np.ndarray, optional
            The reciprocal space vectors as Miller indices. Given as a (N, 3) array.
        isotropic: bool, default True, given in atoms config file, describes wether to use isotropic or aisotropic thermal displacements, 
            currently f' method only configured for isotropic.
        absorption : bool, default False
            If True, includes absorptive scattering components in the calculation.
        absorption_type : {'param', 'constant'}, default 'param'
            - 'param': Uses parameterized values for absorption.
            - 'constant': Computes constant absoprtative potential.
        
        Returns
        -------
        torch.tensor (N_atoms, N_hkls)
            The scattering factors.

        """
        parametrization = validate_parametrization(parametrization)

        # TODO tidy
        Z_unique = np.unique(numbers.detach().cpu().numpy())

        scattering_factors = {Z: parametrization.scattering_factor(Z) for Z in Z_unique}
        f_e = torch.zeros((len(positions), len(g)), dtype=torch.complex128, device=positions.device)

        DWF = self.calculate_dwf_factor(thermal_disps, hkls, reciprocal_cell)

        factor_primes = None
        
        s = g/2  # Convert g to s as per Beanland's definition
        if absorption and absorption_type == "param":
            if not isotropic:
                warnings.warn(
                    "Anisotropic displacements not yet supported for absorptive scattering factors, proceeding with isotropic Debye-Waller absorptive scattering factors",
                    UserWarning
                )

            factor_primes = []
            for i in range(len(positions)):
                # keep Utrace as a tensor, not a detached float
                Utrace = torch.mean(torch.diagonal(thermal_disps[i]))  # differentiable
                f_complex = calc_complex_scattering_factors(
                    s,
                    8 * (np.pi**2) * Utrace,
                    velocity,
                    numbers[i],
                )
                # ensure result is tensor on the same device
                factor_primes.append(torch.as_tensor(f_complex, dtype=torch.complex128, device=positions.device))

        # TODO parallelise this
        for i in range(len(positions)):
            Z = numbers[i]
            o = occupancy[i]
            # assumes that Uij is already mean squared displacement as per cif convention, abtem assumes rmsd
            factor = torch.tensor(scattering_factors[Z.item()](g**2), device=positions.device)
            
            if factor_primes is not None:
                factor = factor + 1j * factor_primes[i]

            f_e[i] = factor * DWF[i] * o

        if cutoff == "taper":
            T = 0.005
            alpha = 1 - 0.05
            cutoff_array = 1 / (1 + np.exp((g / g_max - alpha) / T))
        elif cutoff  == "hard":
            cutoff_array = g <= g_max
        else:
            raise ValueError("cutoff must be 'taper' or 'hard'")
        f_e_cutoff = f_e * torch.tensor(cutoff_array, device=positions.device)
        return f_e_cutoff
    


class ApparentThicknessNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.activate = cfg.activate
        self.num_samples = cfg.num_samples
        self.sample_thickness = cfg.sample_thickness
        self.form = cfg.form
        self.thickness_min = cfg.min_thickness
        self.thickness_max = cfg.max_thickness
        self.layers = nn.Sequential(
            nn.Linear(1, 64, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(64, 64, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(64, 2, dtype=torch.float64)  # Outputs: [mu, log_sigma^2]
        )
        if self.form == 'quadratic':
            # use a quadratic form as the basis
            self.quad_layer = nn.Linear(2, 1, bias=True, dtype=torch.float64)

    def denormalize_thickness(self, thickness):
        """ Denormalize thickness using MinMaxScaler, assumes thickness between [-1, 1] """
        return self.thickness_min + (thickness + 1) * (self.thickness_max - self.thickness_min) / 2

    def forward(self, theta):
        theta = theta.unsqueeze(0).unsqueeze(-1) if theta.dim() == 0 else theta.unsqueeze(-1)
        params = self.layers(theta)  # (1, 2)
        residual_mu = params[:, 0]
        log_sigma = params[:, 1]

        if self.form == 'quadratic':
            theta_features = torch.cat([theta, theta**2], dim=-1)
            b, raw_a = self.quad_layer.weight.squeeze(0)
            c = self.quad_layer.bias.squeeze(0)
            #ensure positive curvature
            a = torch.exp(raw_a)
            quad_mu = b * theta + a * theta**2 + c
            mu = ((quad_mu + residual_mu) / 2).squeeze(-1)
            #print(f'mu: {mu}, residual_mu: {residual_mu}, quad_mu: {quad_mu}')

        elif self.form == 'min_thickness':
            mu = params[:,0] #torch.tanh(params[:,0])#self.thickness_min + (self.thickness_max - self.thickness_min) * torch.sigmoid(params[:, 0])
        else:
            raise ValueError(f"Invalid form for thickness nn {self.form}")
        sigma = torch.exp(log_sigma)
        sigma_min, sigma_max = 1, 200
        sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(sigma)

        mu = self.denormalize_thickness(mu)
        return mu, sigma
    
class BlochNet(nn.Module):
    """
    BlochNet class for calculating the dynamical scattering of a crystal.
    Params:
    - cfg: config object
    - sf_nn: StructureFactorNet object
    - refine_vg: bool, whether to refine Fgb
    """
    def __init__(self, cfg, sf_nn, refine_vg=False):
        super(BlochNet, self).__init__()
        self.cfg = cfg
        self.refine_vg = refine_vg
        self.structure_factor_net = sf_nn
        self.thickness_nn = self.structure_factor_net.thickness_nn
        self.absorption = self.structure_factor_net.absorption
        self.energy = self.structure_factor_net.energy
        self.g_max = self.structure_factor_net.solve_g_max  # beam cutoff
        if "g_max" in cfg and not np.isclose(cfg.g_max, self.g_max):
            raise ValueError(
                f"bloch.g_max ({cfg.g_max}) does not match the solve_g_max ({self.g_max}) the StructureFactorNet was built with"
            )
        self.g_max_refine = cfg.g_max_refine
        self.g_min_refine = cfg.g_min_refine
        if self.refine_vg:
            if cfg.g_max_sf:
                self.g_max_sf = cfg.g_max_sf
            else:
                raise ValueError("g_max_sf not specified, please define in config")
                
        self.cell = self.structure_factor_net.atoms.unit_cell
        self.sg_max = cfg.sg_max
        self.hkl = self.structure_factor_net.hkl.clone()

        self.centering = self.structure_factor_net.centering
        self.device = cfg.device
        
        # compute Fgb from structure factor net
        self.Fgb = self.structure_factor_net()
        self.u0_prime = self.structure_factor_net.U_0_prime
        prefactor =  energy2sigma(self.energy) / (kappa * 1 * energy2wavelength(self.energy) * np.pi)
        self.U0 = torch.abs(self.Fgb[0] * prefactor).detach().cpu().numpy()

        if self.refine_vg:
            self.initial_Fgb = self.Fgb.clone().detach()
            self.Fgb = nn.Parameter(self.Fgb)
            # create mask for Fgb grads that are higher res than we want to refine
            mask = gmax_mask(self.hkl.cpu().numpy(),
                                  self.structure_factor_net.atoms.reciprocal_cell(),
                                  self.g_max_sf)
            # mask2 = torch.abs(self.Fgb) >= 0.04
            # mask2 = torch.tensor(mask2, dtype = torch.bool, device=self.device)
            self.mask = torch.tensor(mask, device=self.device) 
            Fgb_symmetry_constraints_idx_list, phase_diff_list = create_Fgb_symmetry_restraints(
                symmetry_operations=self.structure_factor_net.atoms.spacegroup.get_symop(),
                hkls=self.hkl,
                Fgb=self.Fgb.detach(),    # don't want to differentiate through this
                mask=self.mask,
                tol=1e-4,
            )
            self.mask = self.mask #* mask2
            self.Fgb_symmetry_constraints_idxs = torch.tensor(Fgb_symmetry_constraints_idx_list, dtype=torch.int, device=self.device)
            self.Fgb_phase_diff = torch.tensor(phase_diff_list, dtype=torch.float64, device=self.device) # TODO dtype?
            print(f"Number of Fgb components in resolution limit for refinement: {len(self.Fgb[self.mask])}")
            print(f"Number of Fgb symmetry constraints: {len(self.Fgb_symmetry_constraints_idxs)}")
            # print(f"Fgb_symmetry_constraints_idxs: {self.Fgb_symmetry_constraints_idxs}")
    
    @property
    def wavelength(self) -> float:
        """The wavelength of the electrons [Å]."""
        return energy2wavelength(self.energy)

    @property
    def structure_matrix_nbytes(self) -> int:
        """The number of bytes used by the structure matrix."""
        bytes_per_element = 128 // 8
        return self.num_bloch_waves**2 * bytes_per_element
    
    def forward(self, orientation_matrix=[np.eye(3)], tilts=None, thickness=None, theta=None, j=None):
        """
        List of (hkl, diffraction pattern) tuples for each orientation.
        """
        # mask out Fgb hkls with mask using torch detach addition trick
        #start = time.time()
        #hkl_selected_list = []
        if self.refine_vg:
            Fgb = self.Fgb*self.mask + self.Fgb.detach()*(~self.mask)
        else:
            Fgb = self.Fgb
        
        if self.thickness_nn.activate:
            if theta is None:
                warnings.warn("Theta is None. Expected a valid value for theta.")
            else:
                mu, sigma = self.thickness_nn(theta)
                #print(f'mu: {mu}, sigma: {sigma}')
                if self.thickness_nn.sample_thickness:
                    # sample thicknesses from normal distribution
                    thickness = sample_thicknesses(mu, sigma, self.thickness_nn.num_samples)
                    print(f'sigma:{sigma}')
                else:
                    thickness = [mu]
        # warning for default thickness
        elif thickness is None:
            thickness = [500]
            warnings.warn("Default thickness of 500 Å used, please specify thickness for accurate results.")

        if self.cfg.integrated_intensities:
            # TODO maybe change to warning
            assert tilts is not None, "tilts must be specified for integrated intensities"
            # be careful with rotation mat order, R2 is being applied after rotation_dataset.dataset.rotations, so on left sife of np.dot
            orientation_matrices = [np.dot(R2, orientation_matrix) for R2 in tilts]
        else:
            orientation_matrices = [orientation_matrix]
        results_dataset = DiffractionDataset()

        untilted_cell = np.dot(self.structure_factor_net.atoms.unit_cell, orientation_matrix.T)
        
        #untilted_cell = np.dot(untilted_cell_gt, M.T)
        untilted_reciprocal_cell = reciprocal_cell(untilted_cell)
        for i, tilt in enumerate(orientation_matrices):
            # Apply orientation matrix
            rotated_cell = np.dot(self.structure_factor_net.atoms.unit_cell, tilt.T)
            #rotated_cell = np.dot(rotated_cell, M.T)
            reciprocal_rotated_cell = reciprocal_cell(rotated_cell)
            hkl_mask = filter_reciprocal_space_vectors(
                hkl=self.hkl.detach().cpu().numpy(),    # TODO cuda?
                reciprocal_cell=reciprocal_rotated_cell,
                energy=self.energy,
                sg_max=self.sg_max,
                g_max=self.g_max,
                centering=self.centering,
            )
           
            selected_hkl = self.hkl[hkl_mask].detach().cpu().numpy().astype(int)
            # print(f'selected_hkl: {selected_hkl}')
            #hkl_selected_list.append(selected_hkl)
            #rotated_g_vec = self.hkl @ self.structure_factor_net.atoms.reciprocal_cell()
            # Calculate structure matrix using the rotated cell
            A_rotated = calculate_structure_matrix(
                structure_factor=Fgb,
                hkl=self.structure_factor_net.hkl.detach().cpu().numpy().astype(int), # Original hkls are used, change to ints
                hkl_selected=selected_hkl,              # Masked hkls for this simulation
                cell=rotated_cell,
                energy=self.energy,
                U_0_prime = self.u0_prime,
                gpts=self.structure_factor_net.gpts,
                device=self.device,
                absorption = self.absorption,
                U0=self.U0,
            )
            # Calculate diffraction pattern for the rotated structure matrix
            array = calculate_dynamical_scattering(
                structure_matrix=A_rotated,
                hkl=selected_hkl,
                cell=rotated_cell,
                energy=self.energy,
                thicknesses=thickness,
                device=self.device,
                absorption=self.absorption,
                U0=self.U0,
            )

            reciprocal_lattice_vectors = reciprocal_cell(rotated_cell)
            results_dataset.store_results(torch.abs(array) ** 2, reciprocal_lattice_vectors, selected_hkl, untilted_reciprocal_cell, i)
        results_dataset.thicknesses = thickness
        #np.savez_compressed(f"/scratch-ssd/tiarty/diffBloch/data/urea/synthetic_data/structure_factors/hkl_selected_data_{j}.npz", hkl_selected=np.array(hkl_selected_list, dtype =object), allow_pickle=True)

        #print(print(f"Execution time: {time.time() - start:.2f} seconds"))
        return results_dataset


def calculate_dynamical_scattering(
    structure_matrix: np.ndarray,
    hkl: np.ndarray,
    cell: np.ndarray,
    energy: float,
    thicknesses: Sequence[float],
    device: str = "cpu",
    absorption : bool = False,
    U0: float = 0.0,
) -> torch.tensor:

    """Calculate the dynamical scattering given a structure matrix.

    Parameters
    ----------
    structure_matrix : np.ndarray
        The structure matrix as a (N, N) array.
    hkl : np.ndarray
        The reciprocal space vectors as Miller indices. Given as a (N, 3) array.
    cell : Cell
        The unit cell.
    energy : float
        The energy of the electrons [eV].
    thicknesses : sequence of floats
        The thicknesses of the sample [Å].
    absorption : bool, default False
            If True, includes absorptive scattering components in the calculation.
    Returns
    -------
    list of torch.tensor
        The dynamical scattering as a complex tensor with shape
        (len(thicknesses), len(hkl)).
    """

    Mii = torch.tensor(calculate_M_matrix(hkl, cell, energy, U0=U0), device = device)

    if (absorption):
        v, C_temp = torch.linalg.eig(structure_matrix)
    else:
        #check that Hermitian
        assert torch.allclose(structure_matrix, structure_matrix.conj().T), "structure matrix not hermitian check if absoprtion set in cfg in both bloch and structure factor"
        v, C_temp = torch.linalg.eigh(structure_matrix)

    
    gamma = v / (np.sqrt(1/energy2wavelength(energy)**2 + U0) * 2.0)
    diag = torch.diag(C_temp) / Mii

    diag = diag.to(C_temp.dtype)

    C = fill_diagonal_torch(C_temp, diag)

    if (absorption):
        C_inv = torch.linalg.inv(C_temp)
    else:
        C_inv = torch.conj(C.T)

    initial = np.all(hkl == [0, 0, 0], axis=1).astype(complex)
    initial = torch.tensor(initial, device = device)
    initial = initial.to(C_inv.dtype)

    tensor = torch.zeros((len(thicknesses), len(hkl)), dtype=torch.complex64)
    
    for i, thickness in enumerate(thicknesses):
        
        alpha = torch.matmul(C_inv, initial)
        factor = torch.exp(2.0j * np.pi * thickness * gamma) * alpha
        tensor[i] = torch.matmul(C, factor)
    return tensor
    
def calculate_structure_matrix(
            structure_factor: torch.tensor,
            hkl: np.ndarray,
            hkl_selected: np.ndarray,
            cell: np.ndarray,
            energy: float,
            U_0_prime: float,
            gpts: tuple[int, int, int],
            device: str = "cpu",
            absorption: bool = False,
            U0: float = 0.0,
        ) -> np.ndarray:
            """Calculate the structure matrix for a given set of reciprocal space vectors.

            Parameters
            ----------
            structure_factor : np.ndarray
                The structure factors as a 1D array.
            hkl : np.ndarray
                The reciprocal space vectors as Miller indices corresponding to the structure
                factors. Given as a (N, 3) array.
            hkl_selected : np.ndarray
                The reciprocal space vectors as Miller indices for which the structure matrix is
                calculated. Given as a (N, 3) array.
            cell : Cell
                The unit cell.
            energy : float
                The energy of the electrons [eV].
            gpts : tuple of ints
                The number of grid points in the 3D structure factor.
            absorption : bool, default False
            If True, includes absorptive scattering components in the calculation.
            
            Returns
            -------
            np.ndarray
                The structure matrix.
            """

            g = np.asarray(calculate_g_vec(hkl_selected, cell))
            # print(f'g: {g[0:10]}')
            # print(f'hello')
            Mii = calculate_M_matrix(hkl_selected, cell, energy, U0=U0)
            hkl_selected = np.asarray(hkl_selected)

            gmh = hkl_selected[None] - hkl_selected[:, None]
            gmh = gmh.reshape(-1, 3)
            A = raveled_hkl_to_hkl_torch(structure_factor, hkl, gmh, gpts)
            A = A.reshape((len(hkl_selected),) * 2)
            

            prefactor =  energy2sigma(energy) / (kappa * 1 * energy2wavelength(energy) * np.pi)

            Mii = torch.tensor(Mii, device = device)
            A *= prefactor * Mii[None] * Mii[:, None]
            sg = np.asarray(excitation_errors(g, energy, U0=U0))
            diag = 2 * np.sqrt((1 / (energy2wavelength(energy)))**2 +U0) * sg
            diag = torch.tensor(diag, device = device)



            """ Consider doing the following index:
            U_0 = 7 # need to replace with material specific U_0, maybe add to cfg or can be calculated similarly to how I calculate U_0_prime
            diag = (2 * (np.sqrt((1/energy2wavelength(energy))**2 + U_0)) * sg)
            
            """
            if absorption:
                diag = diag.to(torch.complex128)  # Convert diag to complex128
                
                if U_0_prime is not None: #this checks the type of absorption, instead of including it in cfg directly
                    
                    diag += (1j*U_0_prime)
                else:
                    #print(f"U_0_prime= {U_0_prime}")
                    max_imaginary_off_diag = A.imag[~torch.eye(A.size(0), dtype=bool, device=A.device)].abs().max().item()
                    diag += (1j*1.7*max_imaginary_off_diag) # here the 1.7 was determined empirically. When using paramterized model of absorption we found on average U_0' to be 1.7 times the largest Ug' value. Humphreys (1968) further suggests that V_0' is typically twice V_111‘ and is a reasonable first approximation.  


            diag *= Mii

            diag = diag.to(A.dtype)

            A_filled = fill_diagonal_torch(A, diag)
            A_filled_np = A_filled.detach().cpu().numpy()  # Convert to NumPy for safe iteration
            
        
            return A_filled


def calculate_M_matrix(
    hkl: np.ndarray, cell: np.ndarray, energy: float, U0: float = 0.0
) -> np.ndarray:
    """
    Calculate the diagonal M matrix elements (Mii) for a given set of reciprocal space vectors,
    including correction for the mean inner potential.

    Parameters
    ----------
    hkl : np.ndarray
        Miller indices of shape (N, 3).
    cell : np.ndarray
        Unit cell as a (3, 3) matrix in Å.
    energy : float
        Incident electron energy in eV.
    U0 : float, optional
        Mean inner potential correction in units of 1/Å² (default is 0.0).
        Should be precomputed and passed in explicitly.

    Returns
    -------
    np.ndarray
        Vector of Mii values with shape (N,), dimensionless.
    """
    rc = reciprocal_cell(cell)           # (3, 3)
    g = hkl @ rc                         # (N, 3)

    k0 = -1 / energy2wavelength(energy)   # magnitude of vacuum wavevector (1/Å)
    Kn = np.sqrt(k0**2 + U0)           # corrected wavevector inside crystal (1/Å)

    Mii = 1 / np.sqrt(1 + g[:, 2] / -Kn)
    return Mii
