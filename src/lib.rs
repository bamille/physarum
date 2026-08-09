use anyhow::{Context as _, Result};

use rand::{RngCore, Rng};

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
                label: Some("gpu-sim-course device"),
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


// Static uniform buffer containing camera information + sim params.
//
// One buffer, bound by both the compute passes (as `params`) and the render
// pass (as `camera`). std140-compatible by construction: three vec4s after a
// mat4x4, so every member is already 16-byte aligned and there is no implicit
// padding for WGSL to disagree with.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    /// `proj * view`. World space straight to clip space; the vertex shader
    /// does not need the two separately.
    view_proj: [[f32; 4]; 4],
    /// World-space camera basis, so the quad can be expanded facing the viewer.
    right: [f32; 4],
    up: [f32; 4],
    /// `[agent_radius, speed_scale, _, _]` — see `CameraRig`.
    params: [f32; 4],
}

impl Camera {
    pub fn fixed_camera(
        aspect: f32,
        eye: Vec3,
        target: Vec3,
        fov_y: f32,
        radius: f32,
        speed_scale: f32,
    ) -> Self {
        // Near/far scale with how far away the camera actually is. Hardcoded
        // 0.05..100.0 works for a unit-scale scene and silently clips
        // everything when the world is 100 units across, which ours is.
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
            params: [radius, speed_scale, 0.0, 0.0],
        }
    }

    /// The uniform buffer both the compute and render bind groups point at.
    /// Created outside `Sim` and `Renderer` so neither has to own the other.
    pub fn create_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera + params uniform"),
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
    /// Billboard half-size, in world units. This is the on-screen size of one
    /// agent, and is unrelated to the radius agents are seeded within.
    pub agent_radius: f32,
    pub speed_scale: f32,
}

impl CameraRig {
    /// A head-on camera pulled back far enough to frame a disc of
    /// `world_radius` about the origin, with a bit of margin.
    pub fn framing(world_radius: f32, agent_radius: f32) -> Self {
        let fov_y = 60f32.to_radians();
        let dist = (world_radius * 1.2) / (fov_y * 0.5).tan();
        Self {
            eye: Vec3::new(0.0, 0.0, dist),
            target: Vec3::ZERO,
            fov_y,
            agent_radius,
            speed_scale: 1.0,
        }
    }

    pub fn uniform(&self, aspect: f32) -> Camera {
        Camera::fixed_camera(
            aspect,
            self.eye,
            self.target,
            self.fov_y,
            self.agent_radius,
            self.speed_scale,
        )
    }
}


pub struct Sim {
    agents: Vec<Agent>,
    agent_buffers: [wgpu::Buffer; 2],
    agent_bind_groups: [wgpu::BindGroup; 2], // a->b, b->a
    n_agents: u32,
    iter: u32,
}

impl Sim {
    pub fn new(
        ctx: &GpuContext,
        camera_buf: &wgpu::Buffer,
        n_agents: u32,
        init_rad: f32,
        speed: f32,
    ) -> Self {
        // Create agents
        let mut rng = rand::rng();
        let agents = vec![init_rad; n_agents as usize]
        .into_iter()
        .map(
            |init_rad| {
                Agent::new_with_random_placement(&mut rng, init_rad, speed)
            })
        .collect::<Vec<Agent>>();

        // Create buffer. STORAGE covers both the compute passes (read_write)
        // and the vertex shader reading it back read-only; a storage buffer
        // does not need VERTEX usage to be pulled from a vertex stage.
        let buffer_a = ctx.device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Buffer A"),
                size: std::mem::size_of_val(agents.as_slice()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        let buffer_b = ctx.device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Buffer B"),
                size: std::mem::size_of_val(agents.as_slice()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        // Seed A with the initial placement. B is left zeroed; the first step
        // overwrites it wholesale.
        ctx.queue
            .write_buffer(&buffer_a, 0, bytemuck::cast_slice(agents.as_slice()));

        // Ping-pong buffah
        let bgl = ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("ping pong"),
                entries: &[
                    // Buffer A
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None
                    },
                    // Buffer B
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None
                    },
                    // Params
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None
                    },
                ]
            });

        let bg_ab = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Buf A->B"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buf.as_entire_binding(),
                },
            ]
        });

        let bg_ba = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Buff B-> A"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buf.as_entire_binding(),
                },
            ]
        });

        Sim {
            agents,
            agent_buffers: [buffer_a, buffer_b],
            agent_bind_groups: [bg_ab, bg_ba],
            n_agents,
            iter: 0,
        }
    }

    pub fn agent_buffers(&self) -> &[wgpu::Buffer; 2] {
        &self.agent_buffers
    }

    pub fn agent_bind_groups(&self) -> &[wgpu::BindGroup; 2] {
        &self.agent_bind_groups
    }

    pub fn n_agents(&self) -> u32 {
        self.n_agents
    }

    pub fn agents(&self) -> &[Agent] {
        &self.agents
    }

    /// Which of the two agent buffers currently holds live state — i.e. the
    /// one the renderer should read. Step `iter` binds
    /// `agent_bind_groups[iter % 2]`, which reads slot `iter % 2` and writes
    /// the other, so after incrementing `iter` this still points at the fresh
    /// data.
    pub fn current_slot(&self) -> usize {
        (self.iter % 2) as usize
    }

    pub fn iter(&self) -> u32 {
        self.iter
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Agent {
    pos: [f32; 2],
    heading: [f32; 2],
    speed: f32,
    _pad: f32,
}

impl Agent {
    /// Generate agent placement within circular radius R
    fn new_with_random_placement<R: RngCore>(rng: &mut R, rad: f32, speed: f32) -> Self {
        // `r` first: `get_angle` borrows `rng` uniquely for as long as it lives.
        let r = rng.random_range(0.0..rad);
        let mut get_angle = || {rng.random_range(0.0..std::f32::consts::TAU)};

        let pos = Vec2::from_angle(get_angle()) * r;
        let theta = Vec2::from_angle(get_angle());

        Agent {
            pos: pos.into(),
            heading: theta.into(),
            speed: speed,
            _pad: 0.0,
        }
    }
}


// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Path of the render shader, resolved against the crate root so it does not
/// matter what directory the binary is launched from. Loaded at runtime rather
/// than `include_str!`d so that editing WGSL does not trigger a Rust rebuild.
pub const RENDER_SHADER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/render.wgsl");

/// The agent renderer: one camera-facing quad per agent, no vertex buffer.
///
/// The draw is `draw(0..4, 0..n_agents)` over a triangle strip. There is no
/// per-vertex data at all: the vertex shader gets `@builtin(vertex_index)`
/// 0..4 (which corner of the quad) and `@builtin(instance_index)` (which
/// agent), reads the agent's position out of the storage buffer, and offsets
/// it along the camera's `right`/`up` by `params.x`. That is the "one vertex
/// you write pixels from" — everything else is derived on the GPU.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    rig: CameraRig,
    camera_buf: wgpu::Buffer,
    /// One per agent buffer slot; index with `Sim::current_slot()`.
    bind_groups: [wgpu::BindGroup; 2],
    /// `None` until `shaders/render.wgsl` has entry points. Until then the
    /// pass still runs and clears, which is step 1 of the build order in
    /// docs/02.
    pipeline: Option<wgpu::RenderPipeline>,
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
        agent_buffers: &[wgpu::Buffer; 2],
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

        // group(0): the shared camera/params uniform, plus the agent buffer
        // read-only. Read-only storage is visible to every stage, so the
        // vertex shader can pull straight from the simulation's output with no
        // copy and no vertex buffer.
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = |slot: usize, label: &str| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: agent_buffers[slot].as_entire_binding(),
                    },
                ],
            })
        };
        let bind_groups = [bind_group(0, "render A"), bind_group(1, "render B")];

        let pipeline = Self::build_pipeline(ctx, &bgl, config.format);

        let mut this = Self {
            surface,
            config,
            rig,
            camera_buf,
            bind_groups,
            pipeline,
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

    fn build_pipeline(
        ctx: &GpuContext,
        bgl: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
    ) -> Option<wgpu::RenderPipeline> {
        let source = match std::fs::read_to_string(RENDER_SHADER) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{RENDER_SHADER}: {e} — rendering clear colour only");
                return None;
            }
        };
        // Creating a pipeline against a module with no entry points is a hard
        // validation error, so stay in clear-only mode until the shader is
        // actually written. Ignore `//` comments — the placeholder file
        // documents the interface, `@vertex` and all.
        let code: String = source
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        if !(code.contains("@vertex") && code.contains("@fragment")) {
            eprintln!("{RENDER_SHADER}: no @vertex/@fragment yet — rendering clear colour only");
            return None;
        }

        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("render.wgsl"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render layout"),
                // wgpu 29: `Option<&BindGroupLayout>` per slot, not `&BindGroupLayout`.
                bind_group_layouts: &[Some(bgl)],
                // wgpu 29: push constants are "immediates" now.
                immediate_size: 0,
            });

        Some(
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("agent billboards"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        // No vertex buffers. Everything comes from
                        // vertex_index + instance_index + the storage buffer.
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        // Billboards face the camera by construction and the
                        // corner order flips with the camera basis, so culling
                        // would only ever eat quads.
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
                            // Additive: overlapping agents pile up into
                            // brightness instead of the last one winning. No
                            // depth buffer needed, and draw order stops
                            // mattering.
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::SrcAlpha,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent::OVER,
                            }),
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

    /// Draw the agents in `slot` (see `Sim::current_slot`). Returns `false` if
    /// the frame was skipped because the surface had nothing to give.
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
                label: Some("agents"),
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

            if let Some(pipeline) = &self.pipeline {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.bind_groups[slot], &[]);
                // 4 corners of a triangle strip, one instance per agent.
                pass.draw(0..4, 0..n_agents);
            }
        }

        ctx.queue.submit([encoder.finish()]);
        frame.present();
        true
    }
}
