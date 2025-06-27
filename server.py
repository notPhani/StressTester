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
from backend.main import initiate_board, animate, spawnPlayer

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
