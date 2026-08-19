use std::sync::Arc;
use std::time::Instant;

use airplane::{Camera, CameraRig, GpuContext, Renderer, Sim};

use anyhow::Result;

use glam::Vec2;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

/// Trail map resolution. Also the world size: one cell is one world unit, so
/// sensor distances and speeds are all in cells, and no scale factor has to be
/// kept in sync between the two.
const GRID: [u32; 2] = [2048, 2048];

/// Half-extent of the world, in world units. Matches GRID so that exactly one
/// trail cell lands under one world unit.
fn world_half() -> Vec2 {
    Vec2::new(GRID[0] as f32 * 0.5, GRID[1] as f32 * 0.5)
}

/// 10M over a 4.2M-cell grid, i.e. ~2.4 agents per cell. Below ~0.5 per cell
/// the trails are too sparse to reinforce each other and you get wandering
/// rather than networks; above ~4 every cell saturates and detail washes out.
const N_AGENTS: u32 = 10_000_000;

/// Agents start in a disc of this radius, in cells, headings random. A filled
/// circle gives far more structure than scattering over the whole world, which
/// settles into a boring even mesh (docs/02 §1). Scaled with GRID so the
/// starting blob stays the same fraction of the world.
const INIT_RAD: f32 = 600.0;

/// Billboard half-size for the agent overlay, in world units.
const AGENT_RAD: f32 = 0.8;

struct State {
    window: Arc<Window>,
    ctx: GpuContext,
    sim: Sim,
    renderer: Renderer,
    last_frame: Instant,
}

impl State {
    pub fn new(event_loop: &ActiveEventLoop) -> Result<Self> {
        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("beautiful amazing slime sim")
                .with_inner_size(winit::dpi::LogicalSize::new(900, 900)),
        )?);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
            Box::new(event_loop.owned_display_handle()),
        ));
        let surface = instance.create_surface(window.clone())?;
        let ctx = pollster::block_on(GpuContext::new(
            instance,
            Some(&surface),
            wgpu::Features::empty(),
            wgpu::Features::empty(),
        ))?;

        let sim = Sim::new(&ctx, GRID, N_AGENTS, world_half(), INIT_RAD);

        // The camera uniform is owned by the renderer but created out here, so
        // that neither Sim nor Renderer has to own the other to reach it.
        let camera_buf = Camera::create_buffer(ctx.device());
        let size = window.inner_size();
        let renderer = Renderer::new(
            &ctx,
            surface,
            size.width,
            size.height,
            CameraRig::framing(world_half(), AGENT_RAD),
            camera_buf,
            &sim,
        )?;

        Ok(State {
            window,
            ctx,
            sim,
            renderer,
            last_frame: Instant::now(),
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(&self.ctx, width, height);
    }

    fn render(&mut self) {
        // Clamped so that a stall (dragging the window, waking from sleep)
        // does not teleport every agent across the world in one step.
        let dt = self.last_frame.elapsed().as_secs_f32().min(1.0 / 30.0);
        self.last_frame = Instant::now();

        self.sim.step(&self.ctx, dt);

        self.renderer
            .render(&self.ctx, self.sim.current_slot(), self.sim.n_agents());
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match State::new(event_loop) {
            Ok(s) => {
                s.window.request_redraw();
                self.state = Some(s);
            }
            Err(e) => {
                eprintln!("failed to start: {e:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.physical_key {
                    // Overlay the agents themselves on top of the trail map.
                    PhysicalKey::Code(KeyCode::KeyA) => {
                        state.renderer.show_agents = !state.renderer.show_agents;
                    }
                    // Re-seed the agents and wipe the trail.
                    PhysicalKey::Code(KeyCode::KeyR) => state.sim.reset(&state.ctx),
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default())?;
    Ok(())
}
