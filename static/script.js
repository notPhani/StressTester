const _fps = document.getElementById("fps");
const _userCount = document.getElementById("user-count");
const _particleCount = document.getElementById("particles");

const protocol = window.location.protocol === "https:" ? "wss" : "ws";
const host = window.location.host;
const socket = new WebSocket(`${protocol}://${host}/ws`);

socket.binaryType = "arraybuffer";

socket.onopen = () => {
  socket.send(JSON.stringify({
    width: window.innerWidth,
    height: window.innerHeight
  }));
};

function formatNum(num) {
  if (num > 999) return (num / 1000).toFixed(1) + "K";
  return num;
}

function updateHud(Hud_hash) {
  _fps.innerHTML = `⚡ FPS: ${Hud_hash.fps ?? 0}`;
  _userCount.innerHTML = `👤 User Count: ${formatNum(Hud_hash.userCount ?? 0)}`;
  _particleCount.innerHTML = `⭐ Particle Count: ${formatNum(
    Hud_hash.particleCount ?? 0
  )}`;
}

function parseHexColorParticleBuffer(buffer) {
  const dataView = new DataView(buffer);
  const particleSize = 11;
  const particleCount = buffer.byteLength / particleSize;

  const particles = [];

  for (let i = 0; i < particleCount; i++) {
    const offset = i * particleSize;
    const x = dataView.getFloat32(offset, true);
    const y = dataView.getFloat32(offset + 4, true);
    const r = dataView.getUint8(offset + 8);
    const g = dataView.getUint8(offset + 9);
    const b = dataView.getUint8(offset + 10);

    const hex = `#${[r, g, b].map(c => c.toString(16).padStart(2, '0')).join('')}`;

    particles.push({ x, y, color: hex });
  }

  return particles;
}

function renderParticles(ctx, particles) {
  particles.forEach(p => {
    ctx.shadowColor = p.color;
    ctx.shadowBlur = 10;      // Increase for more glow
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  // Reset glow so HUD text or other things don’t glow too
  ctx.shadowColor = "transparent";
  ctx.shadowBlur = 0;
}

let canvas, ctx;

window.onload = () => {
  canvas = document.getElementById("board");
  ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.scale(dpr, dpr);
  ctx.fillStyle = "#f00";
  ctx.beginPath();
  ctx.arc(50, 50, 10, 0, Math.PI * 2);
  ctx.fill();
};

function parseBinaryPacket(buffer) {
  const dataView = new DataView(buffer);
  const fps = dataView.getFloat32(0, true).toPrecision(2);
  const userCount = dataView.getUint32(4, true);
  const particleCount = dataView.getUint32(8, true);

  const particles = [];
  const particleSize = 11;

  for (let i = 0; i < particleCount; i++) {
    const offset = 12 + i * particleSize;
    const x = dataView.getFloat32(offset, true);
    const y = dataView.getFloat32(offset + 4, true);
    const r = dataView.getUint8(offset + 8);
    const g = dataView.getUint8(offset + 9);
    const b = dataView.getUint8(offset + 10);

    const hex = `#${[r, g, b].map(c => c.toString(16).padStart(2, '0')).join('')}`;

    particles.push({ x, y, color: hex });
  }

  return {
    fps,
    userCount,
    particleCount,
    particles
  };
}

socket.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    const parsed = parseBinaryPacket(event.data);
    console.log("Parsed packet:", parsed);

    // Check if particles exist and are within bounds
    if (parsed.particles.length > 0) {
      const p = parsed.particles[0];
      console.log(`First particle: x=${p.x}, y=${p.y}, color=${p.color}`);
    } else {
      console.warn("No particles received!");
    }

    updateHud({
      fps: parsed.fps,
      userCount: parsed.userCount,
      particleCount: parsed.particleCount
    });

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    renderParticles(ctx, parsed.particles);
  }
};
