use std::sync::Arc;

use airplane::{Camera, CameraRig, GpuContext, Renderer, Sim};

use anyhow::Result;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

const N_AGENTS: u32 = 64 * 8;
const INIT_RAD: f32 = 100.0;
const SPEED: f32 = 5.0;

/// Billboard half-size in world units. `INIT_RAD` is how far agents are
/// scattered; this is how big one of them looks.
const AGENT_RAD: f32 = 0.5;

struct State {
    window: Arc<Window>,
    ctx: GpuContext,
    sim: Sim,
    renderer: Renderer,
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

        // The uniform lives outside both Sim and Renderer: the compute passes
        // want the sim params out of it and the vertex shader wants the camera
        // out of it, and neither should have to own the other to get at it.
        let camera_buf = Camera::create_buffer(ctx.device());
        let sim = Sim::new(&ctx, &camera_buf, N_AGENTS, INIT_RAD, SPEED);

        let size = window.inner_size();
        let renderer = Renderer::new(
            &ctx,
            surface,
            size.width,
            size.height,
            CameraRig::framing(INIT_RAD, AGENT_RAD),
            camera_buf,
            sim.agent_buffers(),
        )?;

        Ok(State {
            window,
            ctx,
            sim,
            renderer,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(&self.ctx, width, height);
    }

    fn render(&mut self) {
        // No sim step yet — the compute passes go here, before the draw.
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
