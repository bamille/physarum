# physarium

A GPU slime mold (Physarum) simulation in Rust + wgpu, set up to be developed
entirely offline.

Nothing is implemented yet — this repository is scaffolding, pinned
dependencies, and notes.

## Quick start

```bash
cargo run
```

That currently does nothing but prove the toolchain works. `cargo build` has
already been run once, so the target directory is warm.

## Why "offline" is the organising constraint

**The vendored tree has been removed** to save 565 MB. Cargo now resolves
through crates.io, backed by the shared package cache in `~/.cargo/registry` —
which already holds every crate in `Cargo.lock`, so `cargo build --offline`
still works on this machine. What you lose is the guarantee: a fresh machine,
or a wiped cache, needs network for the first build. `cargo vendor
--versioned-dirs` brings it back.

Full rustdoc for the whole dependency tree is still pre-built at `target/doc/`
(`cargo doc --open`).

The one thing you cannot do offline is **add a dependency**. Decide now if you
want anything beyond what is in `Cargo.toml`.

Details, and how to repair the setup: [docs/04-offline-workflow.md](docs/04-offline-workflow.md).

## Docs

| | |
|---|---|
| [01-what-can-read-write-what.md](docs/01-what-can-read-write-what.md) | Address spaces, buffer/texture binding matrix, usage flags, atomics, and which things are ordered relative to which. The reference for "wait, can a vertex shader write to that?" |
| [02-slime-mold-architecture.md](docs/02-slime-mold-architecture.md) | The algorithm in full, working starting parameters, resource layout, dispatch shapes, and the order to build it in. |
| [03-wgpu-29-api-deltas.md](docs/03-wgpu-29-api-deltas.md) | What changed in wgpu 29 / egui 0.35 / glam 0.33 so that older tutorials do not compile, plus macOS/Metal notes. |
| [04-offline-workflow.md](docs/04-offline-workflow.md) | What works offline, what does not, how to regenerate `vendor/` (now removed). |
| [05-compiling-and-running.md](docs/05-compiling-and-running.md) | Cargo from the ground up: check vs build vs run, debug vs release and why the profiles are set the way they are, env vars, running tests, reading error output, build times. |

## Relationship to gpu-sim-course

Same pinned versions as `../gpu-sim-course`, deliberately, so the patterns
transfer verbatim: `GpuContext` setup, `ComputeKernel` / `Binding`, `PingPong`
(Lesson 10), the egui instrument panel (Lesson 7), and the colormap + fullscreen
blit (Lesson 4). The course's `common/` crate is *not* a dependency here — this
project is standalone — but it is the reference implementation to crib from.

The load-bearing pin is `wgpu = "=29.0.4"`: egui-wgpu 0.35 depends on
`wgpu ^29.0`, so bumping wgpu alone breaks the UI. Bump both or neither.

## Layout

```
Cargo.toml           pinned deps, commented
.cargo/config.toml   build configuration
src/main.rs          empty
shaders/             WGSL goes here
docs/                the notes above
```
