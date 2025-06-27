# server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import struct
import torch
import time

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




BASE_PARTICLES = 50
MAX_PARTICLES = 1000
GROWTH_RATE = 0.05

# all connected WebSocket objects
connected_users = set()

# track how many particles each user owns (for removal on disconnect)
user_particle_map = {}

# global particle state
positions, charges, velocities = initiate_board(1920, 1080, 50)
colors = torch.empty((50, 3), dtype=torch.uint8)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return FileResponse("index.html")


def compute_num_particles(users: int) -> int:
    threshold_users = MAX_PARTICLES // BASE_PARTICLES

    if users <= threshold_users:
        return users * BASE_PARTICLES
    else:
        extra_users = users - threshold_users
        scaling_factor = 1 + (extra_users * GROWTH_RATE)
        return int(min(MAX_PARTICLES * scaling_factor, 5000))


def generate_user_color():
    while True:
        rgb = torch.randint(0, 255, (1, 3), dtype=torch.uint8)
        if rgb.sum() > 200:    # ensure reasonably bright color
            return rgb



async def send_particles(ws: WebSocket, fps: float):
    num_particles = positions.shape[0]

    # build header
    header = struct.pack(
        "<fII",   # little-endian: float, unsigned int, unsigned int
        fps,
        len(connected_users),
        num_particles
    )

    # build particles
    particle_buf = bytearray()
    for i in range(num_particles):
        x = float(positions[i, 0].item())
        y = float(positions[i, 1].item())
        r, g, b = colors[i].tolist()
        particle_buf.extend(struct.pack("<ffBBB", x, y, r, g, b))

    # send entire buffer
    full_buf = header + particle_buf
    await ws.send_bytes(full_buf)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global positions, velocities, charges, colors

    await websocket.accept()
    connected_users.add(websocket)

    try:
        # Receive init data
        init_data = await websocket.receive_text()
        config = json.loads(init_data)
        width = config.get("width", 1920)
        height = config.get("height", 1080)

        # Compute how many new particles this user should spawn
        user_count = len(connected_users)
        desired_particles = compute_num_particles(user_count)
        spawning_count = desired_particles - positions.shape[0]

        user_color = generate_user_color()
        user_particles = 0

        if spawning_count > 0:
            new_pos, new_vel, new_charges = spawnPlayer(
                positions,
                velocities,
                width,
                height,
                spawning_count
            )

            positions = torch.cat([positions, new_pos], dim=0)
            velocities = torch.cat([velocities, new_vel], dim=0)
            charges = torch.cat([charges, new_charges], dim=0)

            new_colors = user_color.repeat(spawning_count, 1)
            colors = torch.cat([colors, new_colors], dim=0)

            user_particles = spawning_count

        user_particle_map[websocket] = (user_particles, user_color)

        # Simply await forever to keep connection alive
        await websocket.receive_text()  # blocks until client closes socket

    except WebSocketDisconnect:
        pass
    finally:
        connected_users.discard(websocket)

        # Remove particles belonging to disconnected user
        particles_to_remove, user_color = user_particle_map.pop(websocket, (0, None))

        if particles_to_remove > 0:
            mask = ~torch.all(colors == user_color, dim=1)
            positions = positions[mask]
            velocities = velocities[mask]
            charges = charges[mask]
            colors = colors[mask]


async def animation_loop():
    global positions, velocities, charges, colors

    prev_time = time.perf_counter()
    while True:
        curr_time = time.perf_counter()
        dt = curr_time - prev_time
        prev_time = curr_time

        if positions.shape[0] > 0:
            positions, velocities = animate(
                prev_position=positions,
                prev_velocity=velocities
            )

        fps = 1.0 / dt if dt > 0 else 60.0

        for ws in list(connected_users):
            try:
                await send_particles(ws, fps)
            except Exception as e:
                print(f"Error sending to client: {e}")

        await asyncio.sleep(1/60)


@app.on_event("startup")
async def start_loop():
    asyncio.create_task(animation_loop())
