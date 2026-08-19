// Pass 1 — sense, steer, move, deposit.
//
// One invocation per agent. Reads the agent out of `src`, writes the updated
// one into `dst`, and adds to the trail map on the way past.

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
};

struct Agent {
    pos:     vec2<f32>,
    heading: vec2<f32>,   // unit vector
    speed:   f32,         // per-agent multiplier on move_speed
    species: u32,         // which trail channel
};

// Ping-pong: read src, write dst. The Rust side swaps which is which every
// step, so never write to src — this step's src is last step's dst.
@group(0) @binding(0) var<storage, read_write> src: array<Agent>;
@group(0) @binding(1) var<storage, read_write> dst: array<Agent>;
@group(0) @binding(2) var<uniform> params: SimParams;

// Sensed *and* deposited into in this same dispatch. Two agents landing on one
// cell in the same step can lose a deposit to the read-modify-write race, and
// an agent may sense a neighbour's deposit from this step or last step. Both
// are fine here: the diffuse pass smears it out and a one-step-stale read is
// as physically defensible as a fresh one. See docs/02 §3.
@group(0) @binding(3) var<storage, read_write> trail: array<vec4<f32>>;


// --- trail channels ---------------------------------------------------------
//
// Four channels per cell. Right now every agent is species 0, but the wiring
// is per-species so multiple kinds can share one trail map:
//
//   deposit_mask  — which channels this species writes
//   sense_mask    — which channels attract it, and how strongly
//
// Making a sense entry negative is how you get one species repelled by
// another's trail, which is where the multi-species patterns come from.

fn deposit_mask(species: u32) -> vec4<f32> {
    switch species {
        case 0u:  { return vec4<f32>(1.0, 0.0, 0.0, 0.0); }
        case 1u:  { return vec4<f32>(0.0, 1.0, 0.0, 0.0); }
        case 2u:  { return vec4<f32>(0.0, 0.0, 1.0, 0.0); }
        default:  { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    }
}

fn sense_mask(species: u32) -> vec4<f32> {
    // Follow your own channel. Try vec4(1.0, -0.5, 0.0, 0.0) for species 0 to
    // make it avoid species 1.
    return deposit_mask(species);
}


// --- grid -------------------------------------------------------------------

/// World position -> flat trail index, wrapping at the edges so the world is a
/// torus and matches how agents wrap.
fn cell_index(pos: vec2<f32>) -> u32 {
    let size = params.world_half * 2.0;
    let uv = fract((pos + params.world_half) / size); // 0..1, wrapped
    let g = vec2<f32>(params.grid);
    let c = vec2<u32>(clamp(uv * g, vec2<f32>(0.0), g - vec2<f32>(1.0)));
    return c.y * params.grid.x + c.x;
}

fn rotate(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);
}


// --- randomness -------------------------------------------------------------
//
// No RNG state in a shader; hash the invocation index and the frame number
// instead, which gives every agent a different value every step.

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

/// Uniform in [0, 1).
fn rand01(seed: u32) -> f32 {
    return f32(hash_u32(seed)) * 2.3283064365386963e-10;
}


// --- sensing ----------------------------------------------------------------

fn sense(pos: vec2<f32>, dir: vec2<f32>, species: u32) -> f32 {
    let sample_pos = pos + dir * params.sensor_dist;
    return dot(trail[cell_index(sample_pos)], sense_mask(species));
}

/// Pick a turn from the three sensor readings: +1 left, 0 straight, -1 right.
///
/// Rather than "strongest sensor wins", this is a weighted random draw with
/// weights exp(sensitivity * strength) — a softmax. `sensitivity` is then a
/// single knob spanning the whole range of behaviour:
///
///   0        every direction equally likely; agents are a gas
///   ~1-3     follows trails but explores; this is where the networks live
///   large    always takes the strongest sensor, i.e. the classic deterministic
///            rule, and the structure gets brittle and crystalline
///
/// Subtracting the max before exponentiating is the standard softmax trick: it
/// leaves the ratios unchanged (they only depend on differences) while keeping
/// every exponent <= 0, so a strong trail cannot overflow to inf. Without it
/// this blows up as soon as trails accumulate.
fn choose_turn(strengths: vec3<f32>, seed: u32) -> f32 {
    let peak = max(strengths.x, max(strengths.y, strengths.z));
    let w = exp((strengths - vec3<f32>(peak)) * params.sensitivity);

    let total = w.x + w.y + w.z;
    let r = rand01(seed) * total;

    if (r < w.x) {
        return 1.0;           // left
    } else if (r < w.x + w.y) {
        return 0.0;           // straight
    }
    return -1.0;              // right
}


@compute @workgroup_size(64, 1, 1)
fn update_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    // A 1D dispatch caps at 65535 workgroups, i.e. 65535 * 64 = 4.19M agents.
    // Past that the Rust side lays the dispatch out as a rough rectangle, and
    // the flat agent index has to be rebuilt from both axes. Below the cap
    // `row_width` covers the whole dispatch and gid.y is always 0, so this one
    // expression is correct either way.
    let i = gid.y * params.row_width + gid.x;

    // The dispatch is rounded up to whole workgroups, so the tail runs past the
    // end of the array. Without this guard those extra invocations write out of
    // bounds.
    if (i >= params.n_agents) {
        return;
    }

    var agent = src[i];
    let seed = i * 747796405u + params.frame * 2891336453u;

    // 1. Sense — three samples: ahead-left, ahead, ahead-right.
    let left_dir = rotate(agent.heading, params.sensor_angle);
    let right_dir = rotate(agent.heading, -params.sensor_angle);
    let strengths = vec3<f32>(
        sense(agent.pos, left_dir, agent.species),
        sense(agent.pos, agent.heading, agent.species),
        sense(agent.pos, right_dir, agent.species),
    );

    // 2. Steer.
    let turn = choose_turn(strengths, seed);
    agent.heading = normalize(rotate(agent.heading, turn * params.turn_speed * params.dt));

    // 3. Move.
    let step = params.move_speed * agent.speed * params.dt;
    agent.pos = agent.pos + agent.heading * step;

    // Boundary: wrap, so the world is a torus. fract() handles an agent that
    // overshoots by more than one world in a single step, and unlike
    // reflection there is no way to get stuck vibrating against the wall.
    let size = params.world_half * 2.0;
    agent.pos = fract((agent.pos + params.world_half) / size) * size - params.world_half;

    // 4. Deposit.
    let cell = cell_index(agent.pos);
    trail[cell] = trail[cell] + deposit_mask(agent.species) * params.deposit;

    dst[i] = agent;
}
