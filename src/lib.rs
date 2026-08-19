use anyhow::{Context as _, Result};

use glam::{Vec2, Vec3};

pub struct GpuContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {

    pub async fn new(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
        required: wgpu::Features,
        optional: wgpu::Features,
    ) -> Result<Self> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface,
            })
            .await
            .context(
                "no suitable GPU adapter found. On Windows this usually means the Vulkan \
                 loader cannot see your driver — see SETUP.md, and try WGPU_BACKEND=dx12",
            )?;

        let available = adapter.features();
        let missing = required - available;
        anyhow::ensure!(
            missing.is_empty(),
            "adapter {:?} does not support required features: {missing:?}",
            adapter.get_info().name
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("supah nice graphics device"),
                required_features: required | (optional & available),
                // `downlevel_defaults` would cap storage buffers at 128 MiB and
                // workgroup storage at 16 KiB. We are targeting a real desktop
                // GPU, so take what the adapter offers.
                required_limits: adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to create wgpu device")?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}


// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// Camera uniform. std140-compatible by construction: three vec4s after a
/// mat4x4, so every member is already 16-byte aligned and there is no implicit
/// padding for WGSL to disagree with.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    /// `proj * view`. World space straight to clip space; the vertex shader
    /// does not need the two separately.
    view_proj: [[f32; 4]; 4],
    /// World-space camera basis, so the quad can be expanded facing the viewer.
    right: [f32; 4],
    up: [f32; 4],
    /// `[agent_radius, _, _, _]`. Everything sim-side now lives in
    /// `SimParams`; this is purely render state.
    params: [f32; 4],
}

impl Camera {
    pub fn fixed_camera(
        aspect: f32,
        eye: Vec3,
        target: Vec3,
        fov_y: f32,
        agent_radius: f32,
    ) -> Self {
        // Near/far scale with how far away the camera actually is. A hardcoded
        // 0.05..100.0 works for a unit-scale scene and silently clips
        // everything when the world is hundreds of units across, which ours is.
        let dist = (eye - target).length().max(1e-3);
        let proj = glam::camera::rh::proj::directx::perspective(
            fov_y,
            aspect,
            (dist * 0.01).max(1e-3),
            dist * 10.0,
        );
        let view = glam::camera::rh::view::look_at_mat4(eye, target, Vec3::Y);

        // The camera basis, read out of the inverse view matrix (= the
        // camera-to-world transform). Expanding the quad along these — rather
        // than world x and y — is what makes it face the viewer from any angle.
        let inv = view.inverse();
        let right = inv.x_axis.truncate().normalize();
        let up = inv.y_axis.truncate().normalize();

        Camera {
            view_proj: (proj * view).to_cols_array_2d(),
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            params: [agent_radius, 0.0, 0.0, 0.0],
        }
    }

    pub fn create_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: std::mem::size_of::<Camera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

/// CPU-side camera state. `Camera` is the packed GPU view of this, rebuilt
/// whenever the window resizes or a value here changes.
#[derive(Copy, Clone, Debug)]
pub struct CameraRig {
    pub eye: Vec3,
    pub target: Vec3,
    pub fov_y: f32,
    /// Half-extent of the simulated world, in world units. The camera frames
    /// exactly this.
    pub world_half: Vec2,
    /// Billboard half-size for a single agent, in world units.
    pub agent_radius: f32,
}

impl CameraRig {
    /// A head-on camera pulled back so the world rectangle exactly fills the
    /// window vertically.
    ///
    /// Note the margin is 1.0, not 1.2. With a margin the simulation sits as a
    /// smaller square inside the window frame with dead space around it, and
    /// agents wrap at an invisible boundary well inside the glass.
    pub fn framing(world_half: Vec2, agent_radius: f32) -> Self {
        let fov_y = 60f32.to_radians();
        let dist = world_half.y / (fov_y * 0.5).tan();
        Self {
            eye: Vec3::new(0.0, 0.0, dist),
            target: Vec3::ZERO,
            fov_y,
            world_half,
            agent_radius,
        }
    }

    pub fn uniform(&self, aspect: f32) -> Camera {
        Camera::fixed_camera(aspect, self.eye, self.target, self.fov_y, self.agent_radius)
    }
}


// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------

/// The whole character of a slime mold lives in these numbers — see docs/02
/// for values known to work and what each one does to the picture.
///
/// Laid out in four 16-byte rows so the WGSL mirror needs no padding games.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    // row 0
    pub grid: [u32; 2],
    pub n_agents: u32,
    pub frame: u32,
    // row 1
    pub world_half: [f32; 2],
    pub dt: f32,
    /// Radians between the centre sensor and each side sensor.
    pub sensor_angle: f32,
    // row 2
    /// How far ahead the sensors sample, in world units (= trail cells).
    pub sensor_dist: f32,
    /// Radians per second an agent can turn.
    pub turn_speed: f32,
    /// World units per second.
    pub move_speed: f32,
    /// Exponent applied to the sensor readings when picking a direction. 0
    /// makes the choice uniformly random and the agents a gas; large values
    /// make it a hard argmax and the agents brittle. See `choose_turn` in
    /// compute.wgsl.
    pub sensitivity: f32,
    // row 3
    /// Added to the trail cell under the agent each step.
    pub deposit: f32,
    /// Trail is multiplied by `exp(-decay_rate * dt)` each step, so the
    /// half-life is in seconds and does not shift with frame rate.
    pub decay_rate: f32,
    /// How far each step blurs toward the neighbourhood mean, 0..1.
    pub diffuse_rate: f32,
    /// Invocations per row of the agent dispatch, so the shader can rebuild a
    /// flat agent index from a 2D dispatch. See `dispatch_2d`.
    pub row_width: u32,
    // row 4
    /// Radius of the disc agents are seeded into, in world units.
    pub init_rad: f32,
    pub _pad: [f32; 3],
}

impl SimParams {
    pub fn new(grid: [u32; 2], n_agents: u32, world_half: Vec2, init_rad: f32) -> Self {
        Self {
            grid,
            n_agents,
            frame: 0,
            world_half: world_half.into(),
            dt: 0.0,
            sensor_angle: 22.5f32.to_radians(),
            sensor_dist: 1.5,
            turn_speed: 8.0,
            move_speed: 100.0,
            sensitivity: 10.0,
            deposit: 1.0,
            decay_rate: 3.0,
            diffuse_rate: 9.0,
            row_width: dispatch_2d(n_agents.div_ceil(WORKGROUP_SIZE)).0 * WORKGROUP_SIZE,
            init_rad,
            _pad: [0.0; 3],
        }
    }
}

/// Maximum workgroups per dispatch dimension. A hard WebGPU limit, not an
/// adapter one — every backend has it.
pub const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Split a workgroup count across two dimensions when it will not fit in one.
///
/// At `@workgroup_size(64)` a 1D dispatch tops out at 65535 * 64 = 4,193,280
/// agents. Past that the dispatch is laid out as a rough square, and the shader
/// rebuilds the flat index as `gid.y * row_width + gid.x`.
pub fn dispatch_2d(total_groups: u32) -> (u32, u32) {
    if total_groups <= MAX_WORKGROUPS_PER_DIM {
        return (total_groups, 1);
    }
    let x = (total_groups as f64).sqrt().ceil() as u32;
    (x, total_groups.div_ceil(x))
}


/// Read a WGSL file and report whether it declares every tag (`@compute`,
/// `@vertex`, ...) outside of a `//` comment.
///
/// Every pipeline goes through this so that a shader you are midway through
/// writing degrades to "that pass does nothing" plus a message, instead of a
/// validation panic that takes the window with it.
fn load_shader(path: &str, tags: &[&str]) -> Option<String> {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{path}: {e} — that pass is disabled");
            return None;
        }
    };
    let code: String = source
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    for tag in tags {
        if !code.contains(tag) {
            eprintln!("{path}: no {tag} entry point yet — that pass is disabled");
            return None;
        }
    }
    Some(source)
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}


// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

pub const INIT_SHADER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/init.wgsl");
pub const COMPUTE_SHADER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/compute.wgsl");
pub const DIFFUSE_SHADER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/diffuse.wgsl");

/// Must match `@workgroup_size(..)` in compute.wgsl. Nothing checks this for
/// you — get it wrong and you silently under- or over-dispatch.
pub const WORKGROUP_SIZE: u32 = 64;
/// Must match `@workgroup_size(..)` in diffuse.wgsl.
pub const DIFFUSE_WORKGROUP: u32 = 8;

/// Trail channels per cell. Four, because a `vec4<f32>` is the natural GPU
/// unit and costs the same as one float — the spare three are what multiple
/// species will deposit into.
pub const TRAIL_CHANNELS: u64 = 4;

// NOTE: `Sim` deliberately does not hold a `&GpuContext`. `State` in main.rs
// owns the `GpuContext` *and* the `Sim`, so a borrow here would make `State`
// self-referential, which does not compile. The methods take `&GpuContext`
// instead.
pub struct Sim {
    agent_buffers: [wgpu::Buffer; 2],
    /// Trail map, ping-ponged: the diffuse pass is a stencil (each cell reads
    /// its neighbours), so it cannot write in place.
    trail_buffers: [wgpu::Buffer; 2],
    params_buf: wgpu::Buffer,

    agent_bind_groups: [wgpu::BindGroup; 2],
    diffuse_bind_groups: [wgpu::BindGroup; 2],
    /// Seeds agent buffer A. Slot resets to 0 alongside it.
    init_bind_group: wgpu::BindGroup,
    agent_pipeline: Option<wgpu::ComputePipeline>,
    diffuse_pipeline: Option<wgpu::ComputePipeline>,
    init_pipeline: Option<wgpu::ComputePipeline>,

    params: SimParams,
    /// Which half of every ping-pong pair currently holds live state.
    slot: usize,
}

impl Sim {
    pub fn new(
        ctx: &GpuContext,
        grid: [u32; 2],
        n_agents: u32,
        world_half: Vec2,
        init_rad: f32,
    ) -> Self {
        let params = SimParams::new(grid, n_agents, world_half, init_rad);

        // --- agents ---------------------------------------------------------
        // No CPU-side agent array: `init.wgsl` seeds buffer A in a dispatch at
        // the end of this function. At 10M agents the CPU path meant a 240 MB
        // allocation and an upload of the same before the window could open.
        //
        // STORAGE covers both the compute passes (read_write) and the vertex
        // shader reading it back read-only; a storage buffer does not need
        // VERTEX usage to be pulled from a vertex stage.
        let agent_bytes = n_agents as u64 * std::mem::size_of::<Agent>() as u64;
        let agent_buffer = |label: &str| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: agent_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let agent_buffers = [agent_buffer("agents A"), agent_buffer("agents B")];

        // --- trail ----------------------------------------------------------
        let trail_bytes = grid[0] as u64 * grid[1] as u64 * TRAIL_CHANNELS * 4;
        let trail_buffer = |label: &str| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: trail_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        // Both start zeroed, which wgpu guarantees — an empty world.
        let trail_buffers = [trail_buffer("trail A"), trail_buffer("trail B")];

        let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim params"),
            size: std::mem::size_of::<SimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- pass 1: agents -------------------------------------------------
        // Trail is bound read_write in the same dispatch that senses it. That
        // is a deliberate race (docs/02 §2): an agent may or may not see a
        // deposit made by another agent in the same step. The diffuse pass
        // launders it and it is invisible in the output.
        let comp = wgpu::ShaderStages::COMPUTE;
        let agent_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("agent pass bgl"),
                entries: &[
                    storage_entry(0, false, comp), // agents src
                    storage_entry(1, false, comp), // agents dst
                    uniform_entry(2, comp),        // params
                    storage_entry(3, false, comp), // trail: sense + deposit
                ],
            });

        let agent_bind_group = |slot: usize, label: &str| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &agent_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: agent_buffers[slot].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: agent_buffers[1 - slot].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: trail_buffers[slot].as_entire_binding(),
                    },
                ],
            })
        };
        let agent_bind_groups = [
            agent_bind_group(0, "agents A->B"),
            agent_bind_group(1, "agents B->A"),
        ];

        // --- pass 2: diffuse + decay ----------------------------------------
        let diffuse_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("diffuse pass bgl"),
                entries: &[
                    storage_entry(0, true, comp),  // trail src, read-only
                    storage_entry(1, false, comp), // trail dst
                    uniform_entry(2, comp),
                ],
            });

        let diffuse_bind_group = |slot: usize, label: &str| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &diffuse_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: trail_buffers[slot].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: trail_buffers[1 - slot].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let diffuse_bind_groups = [
            diffuse_bind_group(0, "diffuse A->B"),
            diffuse_bind_group(1, "diffuse B->A"),
        ];

        let agent_pipeline = Self::compute_pipeline(
            ctx,
            &agent_bgl,
            COMPUTE_SHADER,
            "update_agents",
            "agent pass",
        );
        let diffuse_pipeline =
            Self::compute_pipeline(ctx, &diffuse_bgl, DIFFUSE_SHADER, "diffuse", "diffuse pass");

        // --- pass 0: seeding ------------------------------------------------
        // Its own layout rather than reusing the agent pass's: seeding needs
        // one writable agent buffer and the params, and nothing else. Always
        // targets buffer A, which is why `reset` also puts `slot` back to 0.
        let init_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("init pass bgl"),
                entries: &[storage_entry(0, false, comp), uniform_entry(1, comp)],
            });

        let init_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("init agents A"),
            layout: &init_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agent_buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let init_pipeline =
            Self::compute_pipeline(ctx, &init_bgl, INIT_SHADER, "init_agents", "init pass");

        let mut sim = Sim {
            agent_buffers,
            trail_buffers,
            params_buf,
            agent_bind_groups,
            diffuse_bind_groups,
            init_bind_group,
            agent_pipeline,
            diffuse_pipeline,
            init_pipeline,
            params,
            slot: 0,
        };
        sim.reset(ctx);
        sim
    }

    /// Seed the agents and clear the trail. Also called from `new`.
    pub fn reset(&mut self, ctx: &GpuContext) {
        self.slot = 0;
        // The seeding shader reads n_agents, init_rad and row_width out of the
        // uniform, so it has to be uploaded before the dispatch, not on the
        // first `step`.
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let Some(pipeline) = &self.init_pipeline else {
            return;
        };

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("seed"),
            });
        for trail in &self.trail_buffers {
            encoder.clear_buffer(trail, 0, None);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("init agents"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.init_bind_group, &[]);
            let (gx, gy) = dispatch_2d(self.params.n_agents.div_ceil(WORKGROUP_SIZE));
            pass.dispatch_workgroups(gx, gy, 1);
        }
        ctx.queue.submit([encoder.finish()]);
    }

    /// Called from `new`, while the bind group layouts are still in scope — a
    /// pipeline layout must be built from the same `bgl` the bind groups were,
    /// or the dispatch is a validation error.
    fn compute_pipeline(
        ctx: &GpuContext,
        bgl: &wgpu::BindGroupLayout,
        path: &str,
        entry_point: &str,
        label: &str,
    ) -> Option<wgpu::ComputePipeline> {
        let source = load_shader(path, &["@compute"])?;

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });

        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        Some(
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        )
    }

    /// One simulation step: sense / steer / move / deposit, then diffuse +
    /// decay.
    ///
    /// Both passes go in one encoder. wgpu inserts the barrier between them,
    /// so the diffuse pass is guaranteed to see every deposit.
    pub fn step(&mut self, ctx: &GpuContext, dt: f32) {
        self.params.dt = dt;
        self.params.frame = self.params.frame.wrapping_add(1);
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let (Some(agent_pipeline), Some(diffuse_pipeline)) =
            (&self.agent_pipeline, &self.diffuse_pipeline)
        else {
            return;
        };

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim step"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update agents"),
                timestamp_writes: None,
            });
            pass.set_pipeline(agent_pipeline);
            pass.set_bind_group(0, &self.agent_bind_groups[self.slot], &[]);
            // The dispatch is rounded up — in both dimensions once it goes 2D —
            // so the shader guards the overhang.
            let (gx, gy) = dispatch_2d(self.params.n_agents.div_ceil(WORKGROUP_SIZE));
            pass.dispatch_workgroups(gx, gy, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("diffuse + decay"),
                timestamp_writes: None,
            });
            pass.set_pipeline(diffuse_pipeline);
            pass.set_bind_group(0, &self.diffuse_bind_groups[self.slot], &[]);
            pass.dispatch_workgroups(
                self.params.grid[0].div_ceil(DIFFUSE_WORKGROUP),
                self.params.grid[1].div_ceil(DIFFUSE_WORKGROUP),
                1,
            );
        }

        ctx.queue.submit([encoder.finish()]);

        // Both pairs flip together: pass 1 wrote agents[1-slot] and pass 2
        // wrote trail[1-slot], so one index keeps tracking both.
        self.slot = 1 - self.slot;
    }

    pub fn agent_buffers(&self) -> &[wgpu::Buffer; 2] {
        &self.agent_buffers
    }

    pub fn trail_buffers(&self) -> &[wgpu::Buffer; 2] {
        &self.trail_buffers
    }

    pub fn params_buffer(&self) -> &wgpu::Buffer {
        &self.params_buf
    }

    pub fn params(&self) -> &SimParams {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut SimParams {
        &mut self.params
    }

    pub fn n_agents(&self) -> u32 {
        self.params.n_agents
    }

    /// Which half of each ping-pong pair holds live state — what the renderer
    /// should read.
    pub fn current_slot(&self) -> usize {
        self.slot
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Agent {
    pos: [f32; 2],
    heading: [f32; 2],
    /// Per-agent multiplier on `SimParams::move_speed`.
    speed: f32,
    /// Which trail channel this agent deposits into and follows. All 0 for
    /// now; this is the hook for multiple species.
    species: u32,
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

pub const TRAIL_SHADER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/trail.wgsl");
pub const RENDER_SHADER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/render.wgsl");

/// Two draws, both vertex-buffer-less:
///
/// 1. The trail map, as a single world-space quad spanning the simulation
///    bounds, transformed by the camera and colour-mapped in the fragment
///    shader. This is the picture — an agent itself is one pixel.
/// 2. Optionally the agents, as camera-facing billboards over the top. Useful
///    for seeing what they are actually doing; press A.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    rig: CameraRig,
    camera_buf: wgpu::Buffer,

    trail_bind_groups: [wgpu::BindGroup; 2],
    trail_pipeline: Option<wgpu::RenderPipeline>,
    agent_bind_groups: [wgpu::BindGroup; 2],
    agent_pipeline: Option<wgpu::RenderPipeline>,

    pub show_agents: bool,
    clear: wgpu::Color,
}

impl Renderer {
    pub fn new(
        ctx: &GpuContext,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
        rig: CameraRig,
        camera_buf: wgpu::Buffer,
        sim: &Sim,
    ) -> Result<Self> {
        let config = surface
            .get_default_config(&ctx.adapter, width.max(1), height.max(1))
            .context("surface is not supported by this adapter")?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            present_mode: wgpu::PresentMode::AutoVsync,
            ..config
        };
        surface.configure(&ctx.device, &config);

        // --- trail pass -----------------------------------------------------
        // Read-only storage is visible to every stage, so the fragment shader
        // reads the trail buffer directly. No texture, no sampler, no format
        // table (docs/02 §2).
        let trail_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("trail render bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT), // camera
                    storage_entry(1, true, wgpu::ShaderStages::FRAGMENT),  // trail
                    uniform_entry(2, wgpu::ShaderStages::VERTEX_FRAGMENT), // sim params
                ],
            });

        let trail_bind_group = |slot: usize, label: &str| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &trail_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sim.trail_buffers()[slot].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: sim.params_buffer().as_entire_binding(),
                    },
                ],
            })
        };
        let trail_bind_groups = [
            trail_bind_group(0, "trail A"),
            trail_bind_group(1, "trail B"),
        ];

        // --- agent pass -----------------------------------------------------
        let agent_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("agent render bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                    storage_entry(1, true, wgpu::ShaderStages::VERTEX),
                ],
            });

        let agent_bind_group = |slot: usize, label: &str| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &agent_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sim.agent_buffers()[slot].as_entire_binding(),
                    },
                ],
            })
        };
        let agent_bind_groups = [
            agent_bind_group(0, "agents A"),
            agent_bind_group(1, "agents B"),
        ];

        // The trail pass covers the whole world quad, so it does not blend.
        // Agents go over the top additively.
        let trail_pipeline =
            Self::render_pipeline(ctx, &trail_bgl, TRAIL_SHADER, config.format, None, "trail");
        let agent_pipeline = Self::render_pipeline(
            ctx,
            &agent_bgl,
            RENDER_SHADER,
            config.format,
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            }),
            "agent billboards",
        );

        let mut this = Self {
            surface,
            config,
            rig,
            camera_buf,
            trail_bind_groups,
            trail_pipeline,
            agent_bind_groups,
            agent_pipeline,
            show_agents: false,
            clear: wgpu::Color {
                r: 0.01,
                g: 0.01,
                b: 0.02,
                a: 1.0,
            },
        };
        this.upload_camera(ctx);
        Ok(this)
    }

    fn render_pipeline(
        ctx: &GpuContext,
        bgl: &wgpu::BindGroupLayout,
        path: &str,
        format: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
        label: &str,
    ) -> Option<wgpu::RenderPipeline> {
        let source = load_shader(path, &["@vertex", "@fragment"])?;

        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                // wgpu 29: `Option<&BindGroupLayout>` per slot.
                bind_group_layouts: &[Some(bgl)],
                // wgpu 29: push constants are "immediates" now.
                immediate_size: 0,
            });

        Some(
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        // No vertex buffers anywhere in this program.
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                }),
        )
    }

    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height.max(1) as f32
    }

    pub fn rig_mut(&mut self) -> &mut CameraRig {
        &mut self.rig
    }

    /// Repack the rig into the uniform buffer. Call after touching the rig.
    pub fn upload_camera(&mut self, ctx: &GpuContext) {
        let camera = self.rig.uniform(self.aspect());
        ctx.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&camera));
    }

    pub fn resize(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return; // minimised; a zero-sized surface is a validation error
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&ctx.device, &self.config);
        self.upload_camera(ctx);
    }

    /// Returns `false` if the frame was skipped because the surface had
    /// nothing to give.
    pub fn render(&mut self, ctx: &GpuContext, slot: usize, n_agents: u32) -> bool {
        // wgpu 29: this is an enum, not a Result.
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&ctx.device, &self.config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return false,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None, // wgpu 29; only meaningful for 3D views
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if let Some(pipeline) = &self.trail_pipeline {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.trail_bind_groups[slot], &[]);
                pass.draw(0..4, 0..1); // one quad, spanning the world
            }

            if self.show_agents {
                if let Some(pipeline) = &self.agent_pipeline {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &self.agent_bind_groups[slot], &[]);
                    // 4 corners of a triangle strip, one instance per agent.
                    pass.draw(0..4, 0..n_agents);
                }
            }
        }

        ctx.queue.submit([encoder.finish()]);
        frame.present();
        true
    }
}
