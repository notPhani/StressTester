import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")
_ = torch.randn(1,device=device)
torch.cuda.synchronize()

def initiate_board(width: float, height: float, num_samples: int) -> tuple:
    positions = torch.rand((num_samples, 2), device=device) * torch.tensor([width, height], device=device)
    charges = torch.where(torch.rand(num_samples, device=device) > 0.5, 1.0, 1.0)
    velocities = torch.rand((num_samples, 2), device=device)
    return positions, charges, velocities

def spawnPlayer(prev_positions,prev_velocities: torch.Tensor, width, height, num_samples, cell_size=10, device=device):
    # Discretize previous positions to grid
    grid_x = (prev_positions[:, 0] // cell_size).long()
    grid_y = (prev_positions[:, 1] // cell_size).long()

    grid_w, grid_h = int(width // cell_size), int(height // cell_size)
    grid_x = torch.clamp(grid_x, 0, grid_w - 1)
    grid_y = torch.clamp(grid_y, 0, grid_h - 1)

    occupancy = torch.zeros((grid_w, grid_h), dtype=torch.bool, device=device)
    occupancy[grid_x, grid_y] = True

    
    # Find empty cells
    empty_cells = torch.nonzero(~occupancy, as_tuple=False)
    n_empty = empty_cells.size(0)
    n_spawn = min(num_samples, n_empty)
    
    if n_spawn == 0:
        return torch.empty((0, 2), device=device)  
    idx = torch.randperm(n_empty, device=device)[:n_spawn]
    selected_cells = empty_cells[idx]

    # Convert to coordinates (cell centers)
    spawn_x = (selected_cells[:, 0].float() + 0.5) * cell_size
    spawn_y = (selected_cells[:, 1].float() + 0.5) * cell_size
    spawn_positions = torch.stack([spawn_x, spawn_y], dim=1)
    charges = torch.where(torch.rand(num_samples, device=device) > 0.5, 1.0, -1.0)
    mean_velocity = prev_velocities.mean(dim=0, keepdim=True)
    velocities = mean_velocity.expand(num_samples, -1).clone()

    
    return spawn_positions,velocities,charges

def animate(
        prev_position: torch.Tensor,
        prev_velocity: torch.Tensor,
        rad_influence: float = 500,
        inv_falloff: float = 0.01,
        force_cap: float = 1e6,
        entrpy_factor: float = 0.9,
        charge: float = 8.99e1,
        boundary: str = 'bounce',
        width=1920,
        height=1080
        ) -> torch.Tensor:
    
    # Vectorized pairwise distance calculation
    diff = prev_position.unsqueeze(1) - prev_position.unsqueeze(0)
    dists_sq = (diff**2).sum(dim=2) + 1e-9
    
    # Vectorized mask creation
    diagonal_mask = ~torch.eye(prev_position.size(0), dtype=torch.bool, device=device)
    radial_mask = dists_sq <= rad_influence**2
    force_mag_val = charge / dists_sq
    falloff_mask = force_mag_val > inv_falloff
    force_cap_mask = force_mag_val < force_cap
    valid_mask = diagonal_mask & radial_mask & falloff_mask & force_cap_mask
    
    # Vectorized force direction calculation
    dists = torch.sqrt(dists_sq)
    force_dir = diff / dists.unsqueeze(-1)
    
    # Vectorized force calculation
    forces = force_mag_val.unsqueeze(-1) * force_dir
    forces = torch.where(valid_mask.unsqueeze(-1), forces, torch.zeros_like(forces))
    net_force = forces.sum(dim=1)
    
    # Vectorized noise and velocity update
    noise = torch.randn_like(net_force) * entrpy_factor
    total_velocity = prev_velocity + net_force 
    
    # Vectorized boundary handling
    if boundary == 'bounce':
        new_position = prev_position + total_velocity * 0.1
        out_of_bounds_x = (new_position[:, 0] < 0) | (new_position[:, 0] > width)
        out_of_bounds_y = (new_position[:, 1] < 0) | (new_position[:, 1] > height)
        total_velocity[out_of_bounds_x, 0] *= -0.8
        total_velocity[out_of_bounds_y, 1] *= -0.8
        new_position = torch.clamp(new_position, 0, width)
        return new_position, total_velocity+noise
    
    return prev_position + total_velocity * 0.1+ noise, total_velocity


