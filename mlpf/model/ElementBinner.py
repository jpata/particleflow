import torch
import torch.nn as nn
import math

class ElementBinner(nn.Module):
    """
    Bins input elements based on their eta and phi coordinates.

    Input features are assumed to have:
    - eta at index 2
    - sin(phi) at index 3
    - cos(phi) at index 4
    """
    def __init__(self, eta_bin_edges, phi_bin_edges, max_elems_per_bin: int):
        """
        Args:
            eta_bin_edges (list or tuple): Defines the edges of eta bins.
                e.g., [-2.5, -1.0, 0.0, 1.0, 2.5] defines 4 eta bins.
            phi_bin_edges (list or tuple): Defines the edges of phi bins.
                e.g., [-pi, -pi/2, 0, pi/2, pi] defines 4 phi bins.
                Phi values are typically in [-pi, pi].
            max_elems_per_bin (int): The maximum number of elements to store in each
                                     eta-phi bin. Elements beyond this count will be
                                     truncated. If a bin has fewer elements, it will
                                     be padded with zeros.
        """
        super().__init__()

        if not isinstance(eta_bin_edges, (list, tuple)) or len(eta_bin_edges) < 2:
            raise ValueError("eta_bin_edges must be a list or tuple with at least 2 values to define bins.")
        if not isinstance(phi_bin_edges, (list, tuple)) or len(phi_bin_edges) < 2:
            raise ValueError("phi_bin_edges must be a list or tuple with at least 2 values to define bins.")
        if not isinstance(max_elems_per_bin, int) or max_elems_per_bin <= 0:
            raise ValueError("max_elems_per_bin must be a positive integer.")

        self.max_elems_per_bin = max_elems_per_bin
        self.num_eta_bins = len(eta_bin_edges) - 1
        self.num_phi_bins = len(phi_bin_edges) - 1

        # Register bin edges as buffers.
        # search_edges are the internal boundaries for torch.searchsorted.
        # If eta_bin_edges = [e0, e1, e2, e3] (3 bins), search_edges = [e1, e2].
        # searchsorted will return indices 0, 1, 2.
        self.register_buffer("eta_search_edges", torch.tensor(eta_bin_edges[1:-1], dtype=torch.float32))
        self.register_buffer("phi_search_edges", torch.tensor(phi_bin_edges[1:-1], dtype=torch.float32))

        # Store all edges for reference if needed (e.g. for clamping, though searchsorted handles out-of-bounds)
        self.register_buffer("eta_all_edges", torch.tensor(eta_bin_edges, dtype=torch.float32))
        self.register_buffer("phi_all_edges", torch.tensor(phi_bin_edges, dtype=torch.float32))


    def forward(self, x_features: torch.Tensor, x_coords: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x_features (torch.Tensor): Input feature tensor of shape (batch, num_elems, input_feature_dim).
            x_coords (torch.Tensor): Tensor with coordinates for binning, shape (batch, num_elems, 3),
                                     expected order: [eta, sin_phi, cos_phi].
            mask (torch.Tensor, optional): Boolean tensor of shape (batch, num_elems)
                indicating valid elements. If None, all elements are considered valid.

        Returns:
            torch.Tensor: Binned elements of shape
                (batch, num_eta_bins, num_phi_bins, max_elems_per_bin, input_feature_dim).
            torch.Tensor: Mask for the binned elements, shape
                (batch, num_eta_bins, num_phi_bins, max_elems_per_bin).
            tuple[torch.Tensor, torch.Tensor]: Original batch and element indices of the kept elements.
        """
        B, N, F = x_features.shape
        device = x_features.device
        dtype = x_features.dtype

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        if x_coords.shape[0] != B or x_coords.shape[1] != N or x_coords.shape[2] != 3:
            raise ValueError(f"x_coords shape mismatch. Expected ({B}, {N}, 3), got {x_coords.shape}")
        
        # Extract relevant features
        eta_vals = x_coords[..., 0]       # (B, N)
        sin_phi_vals = x_coords[..., 1]   # (B, N)
        cos_phi_vals = x_coords[..., 2]   # (B, N)
        phi_vals = torch.atan2(sin_phi_vals, cos_phi_vals) # (B, N)

        # Determine bin indices for each element
        # torch.searchsorted returns indices from 0 to len(search_edges).
        # If value < search_edges[0] -> 0
        # If search_edges[k-1] <= value < search_edges[k] -> k
        # If value >= search_edges[-1] -> len(search_edges)
        # This directly gives 0 to num_bins-1, which is what we want.
        eta_bin_ids = torch.searchsorted(self.eta_search_edges, eta_vals.contiguous(), right=False) # (B, N)
        phi_bin_ids = torch.searchsorted(self.phi_search_edges, phi_vals.contiguous(), right=False) # (B, N)

        # Flatten active elements and their corresponding bin/batch information
        active_x_features = x_features[mask]  # (num_active_total, F)
        if active_x_features.numel() == 0: # Handle case with no active elements
            output_tensor = torch.zeros(B, self.num_eta_bins, self.num_phi_bins, self.max_elems_per_bin, F, device=device, dtype=x_features.dtype)
            output_binned_mask = torch.zeros(B, self.num_eta_bins, self.num_phi_bins, self.max_elems_per_bin, device=device, dtype=torch.bool)
            empty_indices = torch.empty(0, dtype=torch.long, device=device)
            return output_tensor, output_binned_mask, (empty_indices, empty_indices)

        active_eta_bin_ids = eta_bin_ids[mask]    # (num_active_total)
        active_phi_bin_ids = phi_bin_ids[mask]    # (num_active_total)

        # Create batch indices for active elements
        batch_indices_for_active = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, N)[mask] # (num_active_total)
        # Create original element indices (within N) for active elements
        element_indices_in_N_for_active = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, N)[mask] # (num_active_total)

        # Create a global group ID for each active element. This ID uniquely identifies
        # which (batch, eta_bin, phi_bin) an element belongs to.
        # This flattens the batch and eta/phi bin dimensions into a single dimension for easier processing.
        # Max value of this ID will be B * num_eta_bins * num_phi_bins - 1
        group_ids = batch_indices_for_active * (self.num_eta_bins * self.num_phi_bins) + \
                    active_eta_bin_ids * self.num_phi_bins + \
                    active_phi_bin_ids

        num_total_bins_in_batch = B * self.num_eta_bins * self.num_phi_bins

        # Initialize output tensors (flattened along batch and bin dimensions first)
        output_flat_tensor = torch.zeros(num_total_bins_in_batch, self.max_elems_per_bin, F, device=device, dtype=x_features.dtype)
        output_flat_binned_mask = torch.zeros(num_total_bins_in_batch, self.max_elems_per_bin, device=device, dtype=torch.bool)

        # Sort active elements by their group_ids to process them group-wise
        sorted_perm = torch.argsort(group_ids)
        sorted_group_ids = group_ids[sorted_perm]
        sorted_active_x_features = active_x_features[sorted_perm]

        # Generate within-group indices (0, 1, 2, ...) for elements in sorted_group_ids
        # This is a vectorized way to do a "cumulative count" within each group.
        # 1. Mark the start of each new group in the sorted list
        is_new_group = torch.cat((torch.tensor([True], device=device), sorted_group_ids[1:] != sorted_group_ids[:-1]))
        # 2. Create a global arange
        group_indices_arange = torch.arange(len(sorted_group_ids), device=device)
        # 3. For elements starting a new group, their value in group_start_markers is their global index. Others are 0.
        group_start_markers = torch.where(is_new_group, group_indices_arange, torch.tensor(0, device=device, dtype=group_indices_arange.dtype))
        # 4. Propagate the start index of each group forward using cummax.
        #    This gives, for each element, the global index of the start of its group.
        group_start_global_indices = torch.cummax(group_start_markers, dim=0).values
        # 5. Subtracting this from the global arange gives the within-group index.
        within_group_indices = group_indices_arange - group_start_global_indices

        # Filter elements that are within the max_elems_per_bin limit for their group
        keep_mask = within_group_indices < self.max_elems_per_bin

        final_group_ids = sorted_group_ids[keep_mask]
        final_within_group_indices = within_group_indices[keep_mask]
        final_active_x_features_kept = sorted_active_x_features[keep_mask]
        
        # Get original batch and element indices for the elements that were kept
        final_batch_indices_kept = batch_indices_for_active[sorted_perm][keep_mask]
        final_element_indices_in_N_kept = element_indices_in_N_for_active[sorted_perm][keep_mask]
        
        # Scatter the final active elements into the output tensor
        output_flat_tensor[final_group_ids, final_within_group_indices, :] = final_active_x_features_kept
        output_flat_binned_mask[final_group_ids, final_within_group_indices] = True

        # Reshape the flat output tensors to the desired output shape
        output_tensor = output_flat_tensor.view(
            B, self.num_eta_bins, self.num_phi_bins, self.max_elems_per_bin, F
        )
        output_binned_mask = output_flat_binned_mask.view(
            B, self.num_eta_bins, self.num_phi_bins, self.max_elems_per_bin
        )

        return output_tensor, output_binned_mask, (final_batch_indices_kept, final_element_indices_in_N_kept)
