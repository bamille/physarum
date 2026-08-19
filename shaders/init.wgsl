// Pass 0 — agent seeding.
//
// Fills the agent buffer with a random placement, on the GPU. Same
// distribution the CPU version used to produce: uniform over a disc of radius
// `init_rad`, with an independent random heading.
//
// This runs once at startup (and again on reset). Seeding 10M agents CPU-side
// meant a 240 MB allocation, 20M calls into `rand`, and a PCIe upload before
// the window could even appear; here it is one dispatch that never leaves the
// GPU.

struct SimParams {
    grid:         vec2<u32>,
    n_agents:     u32,
    frame:        u32,

    world_half:   vec2<f32>,
    dt:           f32,
    sensor_angle: f32,

    sensor_dist:  f32,
    turn_speed:   f32,
    move_speed:   f32,
    sensitivity:  f32,

    deposit:      f32,
    decay_rate:   f32,
    diffuse_rate: f32,
    row_width:    u32,

    init_rad:     f32,
    pad0:         f32,
    pad1:         f32,
    pad2:         f32,
};

struct Agent {
    pos:     vec2<f32>,
    heading: vec2<f32>,
    speed:   f32,
    species: u32,
};

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var<uniform> params: SimParams;

// Same hash as compute.wgsl. Kept duplicated rather than shared because WGSL
// has no include mechanism — if you change one, change both.
fn hash_u32(x: u32) -> u32 {
    var v = x;
    v ^= v >> 17u;
    v *= 0xed5ad4bbu;
    v ^= v >> 11u;
    v *= 0xac4c1b51u;
    v ^= v >> 15u;
    v *= 0x31848babu;
    v ^= v >> 14u;
    return v;
}

fn rand01(seed: u32) -> f32 {
    return f32(hash_u32(seed)) * 2.3283064365386963e-10;
}

const TAU: f32 = 6.28318530718;

@compute @workgroup_size(64, 1, 1)
fn init_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Same 2D dispatch reconstruction as the update pass — 10M agents does not
    // fit in one dimension either.
    let i = gid.y * params.row_width + gid.x;
    if (i >= params.n_agents) {
        return;
    }

    // Three independent streams from one index. Hashing i with three different
    // odd multipliers is what keeps the radius, the position angle and the
    // heading from being correlated with each other — reusing one hash would
    // put every agent's heading in lockstep with where it spawned.
    let seed = i ^ (params.frame * 0x9e3779b9u);
    let u_radius = rand01(seed * 747796405u + 1u);
    let u_pos = rand01(seed * 2891336453u + 2u);
    let u_head = rand01(seed * 3266489917u + 3u);

    // sqrt spreads agents evenly over the disc. Without it density piles up at
    // the centre, because a ring's area grows with r — the same reason a
    // dartboard's outer rings are bigger.
    let r = params.init_rad * sqrt(u_radius);
    let theta = u_pos * TAU;
    let heading_angle = u_head * TAU;

    var agent: Agent;
    agent.pos = vec2<f32>(cos(theta), sin(theta)) * r;
    agent.heading = vec2<f32>(-cos(heading_angle), -sin(heading_angle));
    agent.speed = 1.0;
    agent.species = 0u;

    agents[i] = agent;
}
