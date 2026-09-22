import numpy as np
import torch

import matplotlib.pyplot as plt

from abtem.core.energy import energy2wavelength

from diffBloch.utils import excitation_errors
import torch
import torch.nn.functional as F
class DiffractionDataset:
    def __init__(self):
        # List to store diffraction patterns, reciprocal lattice vectors, and selected_hkl
        self.diffraction_intensities = []
        self.reciprocal_lattice_vectors = []
        self.selected_hkls = []
        # Attributes for storing the first and last rotated cell
        self.first_rotated_cell = None
        self.last_rotated_cell = None
        self.thicknesses = None

        #corresponds to the cell that Klar et al would describe as being 
        #in the average goniometer postions, i.e. no rotation on either side
        self.untilted_reciprocal_cell = None

    def store_results(self, diffraction_intensities, reciprocal_lattice_vectors, selected_hkl, untilted_reciprocal_cell, index):
        # Ensure the diffraction pattern is stored as a tensor without breaking the computational graph
        self.diffraction_intensities.append(diffraction_intensities)
        # Store reciprocal lattice vectors and selected hkl
        self.reciprocal_lattice_vectors.append(reciprocal_lattice_vectors)
        self.selected_hkls.append(selected_hkl)
        
        # Store the first rotated cell
        if index == 0:
            self.first_rotated_cell = reciprocal_lattice_vectors
        
        # Update the last rotated cell
        self.last_rotated_cell = reciprocal_lattice_vectors
        self.untilted_reciprocal_cell = untilted_reciprocal_cell

    def get_dynamical_intensities(self):
        # Return the stored diffraction patterns as a single tensor for easier processing
        return self.diffraction_intensities

    def get_reciprocal_lattice_vectors(self):
        # Return the stored reciprocal lattice vectors as a list
        return self.reciprocal_lattice_vectors

    def get_selected_hkls(self):
        # Return the stored selected_hkl as a list
        return self.selected_hkls

    def get_first_rotated_cell(self):
        return self.first_rotated_cell

    def get_last_rotated_cell(self):
        return self.last_rotated_cell
    
    def get_k(self, hkls, reciprocal_lattice_vectors):
        return hkls @ reciprocal_lattice_vectors
    

    def get_unique_hkls(self):
        """
        find array of all hkls that appear in experiment
        """
        # Stack all hkl arrays together
        combined_hkls = np.vstack(self.get_selected_hkls())
        
        # Get unique rows
        unique_hkls = np.unique(combined_hkls, axis=0)
        
        return unique_hkls
    
    def get_integrated_intensities(self, mosaicity=None):
        """
        Integrate intensities across tilts for each unique hkl, but keep the intensities separated by thickness.
        
        This version operates on the hkl_intensity_map where each hkl maps to a list of intensities across tilts for multiple thicknesses.
        
        :return: A tuple of stacked integrated intensities across all hkls and the corresponding hkls.
        """
        all_hkls = self.collect_unique_hkls()

        # Create the hkl_intensity_map with intensities for each thickness
        hkl_intensity_map = self.create_hkl_intensity_map(all_hkls, mosaicity)
        
        hkl_to_intensity = {}

        # Iterate over the hkls and their corresponding intensity lists (one list per thickness)
        for hkl, thickness_intensity_lists in hkl_intensity_map.items():
            # List to hold the integrated intensities for each thickness
            integrated_intensity_per_thickness = []

            # Sum the intensities across tilts (0-th dimension) for each thickness
            for intensity_list in thickness_intensity_lists:
                # Stack the intensities and sum across tilts
                integrated_intensity = torch.sum(intensity_list)
                integrated_intensity_per_thickness.append(integrated_intensity)

            # Store the integrated intensities per thickness for the current hkl
            hkl_to_intensity[hkl] = torch.stack(integrated_intensity_per_thickness)

        # Prepare the result in the same format as before: a list of stacked intensities and hkls
        combined_filtered_intensities = [hkl_to_intensity[hkl] for hkl in hkl_to_intensity.keys()]
        combined_filtered_hkls = list(hkl_to_intensity.keys())

        return torch.stack(combined_filtered_intensities).T, combined_filtered_hkls

    
    def filter_hkls(self, energy, rsg, dsg, semiangle):
        """
        filters simulated intensities based on filters defined in
        Klar et al. 2023. https://www.nature.com/articles/s41557-023-01186-1
        """
        #k's at avg goniometer position
        unique_hkls = self.get_unique_hkls()
        k_avg_goni_pos = self.get_k(unique_hkls, self.untilted_reciprocal_cell)
        
        sg_max = np.linalg.norm(k_avg_goni_pos[:, 1:], axis = 1) * np.deg2rad(semiangle)
        
        d_ewald = np.abs(excitation_errors(k_avg_goni_pos, energy))
        mask1 = d_ewald / sg_max < rsg

        mask2 = (sg_max - d_ewald) > dsg

        combined_mask = mask1 & mask2
        filtered_hkls = unique_hkls[combined_mask]
        filtered_sg_max = sg_max[combined_mask]

        self.filtered_hkls = []
        self.filtered_intensities = []

        for intensities, hkls in zip(self.diffraction_intensities, self.selected_hkls):
            # Find rows of hkls that are in filtered_hkls (row-by-row comparison)
            matching_mask = np.array([np.any(np.all(hkl == filtered_hkls, axis=1)) for hkl in hkls])
            
            filtered_intensities = intensities[:, matching_mask]

            # Append only the filtered hkls and their corresponding intensities
            self.filtered_hkls.append(hkls[matching_mask])
            self.filtered_intensities.append(filtered_intensities)
            
    def compare_experimental_simulated_data(self, exp_intensities_dict, hkls, simulated_intensities):
        """
        TODO refactor this to be more readable because it is currently horrific
        """
        hkls_dict = {hkl: simulated_intensity for hkl, simulated_intensity in zip(hkls, simulated_intensities)}

        matched_dict = {}
        for entry in exp_intensities_dict:
            hkl = entry["hkl"]
            experimental_intensity = entry["intensity"]
            experimental_sigma = entry["sigma"]
            if experimental_intensity < 0.01 * experimental_sigma or experimental_intensity < 0:
                experimental_intensity = 0.0

            # Check if the hkl is present in the simulated dictionary
            sim_int = hkls_dict.get(hkl, None)        
            if sim_int is not None:
                matched_dict[hkl] = {
                    "experimental": experimental_intensity,
                    "simulated": sim_int,
                    "sigma": experimental_sigma,
                }
        experimental_intensities = []
        experimental_sigmas = []
        hkl_list = []
        filtered_sim_ints = []
        for hkl, data in matched_dict.items():
            experimental_intensity = data["experimental"]
            experimental_sigma = data["sigma"]
            sim_intensity = data["simulated"]
            hkl_list.append(hkl)
            experimental_intensities.append(experimental_intensity)
            experimental_sigmas.append(experimental_sigma)
            filtered_sim_ints.append(sim_intensity)
        experimental_intensities = torch.tensor(
            np.array(experimental_intensities),
            device=simulated_intensities.device,
            dtype=simulated_intensities.dtype,
        )
        experimental_sigmas = torch.tensor(
            np.array(experimental_sigmas),
            device=simulated_intensities.device,
            dtype=simulated_intensities.dtype,
        )
        return (
            experimental_intensities,
            experimental_sigmas,
            torch.stack(filtered_sim_ints),
            hkl_list,
        )

    def collect_unique_hkls(self):
        """
        Collect all unique hkls across the dataset and return a sorted list of unique hkls.
        """
        all_hkls_set = set()

        # Iterate through filtered hkls to collect all unique hkls
        for hkls in self.filtered_hkls:
            for hkl in hkls:
                all_hkls_set.add(tuple(hkl))  # Convert hkl to tuple for hashability

        # Convert the set to a sorted list to ensure consistency
        return sorted(all_hkls_set)


    def moving_average(self, intensities, window_size):
        """
        Compute the moving average over a list of intensities using unfold.
        
        :param intensities: A torch tensor of intensities.
        :param window_size: The size of the moving average window.
        :return: A torch tensor of averaged intensities.
        """
        # Use unfold to create sliding windows of size `window_size`
        unfolded = intensities.unfold(0, window_size, 1)
        
        # Compute the mean across each window
        moving_avg = unfolded.mean(dim=-1)

        # Pad the result to match the original size by adding values at the start and end
        pad_size = (window_size - 1) // 2
        return F.pad(moving_avg, (pad_size, pad_size), mode='constant', value=0)


    def create_hkl_intensity_map(self, all_hkls, mosaicity=None):
        """
        Create a mapping between each unique hkl and its corresponding intensities across the dataset,
        supporting multiple thicknesses, with an optional moving average applied to the intensities across tilts.

        :param all_hkls: A list of all unique hkls.
        :return: A dictionary mapping each hkl to a list of averaged intensities per thickness.
        """
        hkl_intensity_map = {hkl: [[] for _ in range(len(self.thicknesses))] for hkl in all_hkls}

        for hkls, intensities in zip(self.filtered_hkls, self.filtered_intensities):
            if intensities.ndim == 1:  
                intensities = intensities.unsqueeze(0)  # Reshape for single thickness case

            num_thicknesses, num_hkls = intensities.shape

            for thickness_idx in range(num_thicknesses):
                temp_hkl_intensity = {tuple(hkl): intensity for hkl, intensity in zip(hkls, intensities[thickness_idx])}

                for hkl in all_hkls:
                    if hkl in temp_hkl_intensity:
                        hkl_intensity_map[hkl][thickness_idx].append(temp_hkl_intensity[hkl])
                    else:
                        hkl_intensity_map[hkl][thickness_idx].append(torch.tensor(0.0))

        #apply moving averages if accounting for mosaicity
        for hkl, thickness_intensity_lists in hkl_intensity_map.items():
            for thickness_idx in range(len(thickness_intensity_lists)):
                intensity_list = thickness_intensity_lists[thickness_idx]
                if len(intensity_list) > 0:  # Only apply torch.stack if we have intensities
                    stacked_intensity = torch.stack(intensity_list)

                    if mosaicity:
                        hkl_intensity_map[hkl][thickness_idx] = self.moving_average(stacked_intensity, window_size=5)
                    else:
                        hkl_intensity_map[hkl][thickness_idx] = stacked_intensity

        return hkl_intensity_map





    def create_plots(self, hkl_intensity_map, integration_semiangle, rocking_curve_sampling, max_plots=50):
        """
        Plot rocking curves for each hkl.

        :param hkl_intensity_map: Dictionary mapping hkl to a list of intensities.
        :param max_plots: Maximum number of plots to display, default is 50.
        """

        x_data = np.linspace(-integration_semiangle, integration_semiangle, rocking_curve_sampling)

        plot_count = 0

        for hkl, intensity_list in hkl_intensity_map.items():
            print(intensity_list[0].shape)
            if plot_count >= max_plots:
                break  # Stop after reaching max_plots

            # Plot the intensities for each hkl
            plt.figure()
            plt.plot(x_data, intensity_list[0].detach().cpu().numpy(), marker='o', label=f'hkl: {hkl}')
            plt.xlabel('x axis (arbitrary)')
            plt.ylabel('Intensity')
            plt.title(f'Intensity plot for hkl: {hkl}')
            plt.legend()
            plt.grid(True)
            plt.show()

            plot_count += 1

    def plot_rocking_curves(self, integration_semiangle, rocking_curve_sampling, mosaicity=None, max_plots=50):
        """
        Main method to collect hkls, map intensities, and plot rocking curves.
        """
        # Step 1: Collect all unique hkls
        all_hkls = self.collect_unique_hkls()

        # Step 2: Create the hkl to intensity map
        hkl_intensity_map = self.create_hkl_intensity_map(all_hkls, mosaicity)

        # Step 3: Plot the rocking curves
        self.create_plots(hkl_intensity_map, integration_semiangle, rocking_curve_sampling, max_plots=max_plots)




        

    