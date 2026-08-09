# Compiling and running

A practical cargo guide for this project. Everything here works offline.

---

## 1. The four commands you will actually use

| Command | What it does | When |
|---|---|---|
| `cargo check` | Type-checks. Does **not** link or produce a binary. | The inner loop. 3–10× faster than `build`. |
| `cargo build` | Compiles and links `target/debug/airplane`. | When you want to run it. |
| `cargo run` | `build`, then runs the binary. | Almost always the one you want. |
| `cargo test` | Builds test binaries and runs them. | See §5. |

`cargo run` implies `cargo build` implies `cargo check`. Running `cargo build`
before `cargo run` is wasted keystrokes, not a wasted compile — cargo does not
redo work.

**Use `cargo check` while you are fighting the borrow checker.** Linking is the
slow half of a build, and a program that does not type-check does not need to be
linked. Switch to `cargo run` when the errors stop.

---

## 2. Debug vs release

```bash
cargo run              # debug profile   -> target/debug/airplane
cargo run --release    # release profile -> target/release/airplane
```

This project's `Cargo.toml` deliberately blurs the usual line:

```toml
[profile.dev.package."*"]
opt-level = 3          # dependencies: fully optimised even in debug

[profile.dev]
opt-level = 1          # your code: lightly optimised
```

Why: in a GPU simulation, the CPU-side work is mostly wgpu calls, and
unoptimised wgpu is genuinely slow — slow enough that an unoptimised debug build
gives you misleading frame times. Dependencies compile once and then sit in the
cache, so paying `opt-level = 3` for them costs you nothing on rebuilds.

Your own code stays at `opt-level = 1`: fast enough to be usable, and it keeps
line numbers, variable inspection, and panic backtraces intact.

**Practical rule:** develop in debug, and only reach for `--release` when you
are measuring performance or pushing agent counts into the millions. `--release`
also rebuilds every dependency from scratch the first time (several minutes),
into a separate `target/release/` tree.

The GPU does not care about your Rust profile. Shader compilation happens at
runtime, and a WGSL kernel runs at the same speed under `cargo run` and
`cargo run --release`. If your simulation is slow, the profile is usually not
the reason — the dispatch is.

---

## 3. Offline flags

`.cargo/config.toml` already sets `net.offline = true`, so all of the above
work with no network and no extra flags. Adding `--offline` explicitly is
harmless and makes the intent obvious:

```bash
cargo build --offline
```

If you ever see cargo attempt to reach crates.io, something has overridden the
config — check you are running from inside the project directory.
`.cargo/config.toml` is found by walking **up** from the current directory, so
running cargo from `/` or from a sibling project does not pick it up.

---

## 4. Running it

```bash
cargo run
```

Useful environment variables, none of which need a rebuild:

```bash
RUST_BACKTRACE=1 cargo run          # backtrace on panic
RUST_BACKTRACE=full cargo run       # including std frames
RUST_LOG=wgpu_core=warn cargo run   # wgpu validation chatter
RUST_LOG=airplane=debug cargo run   # your own env_logger output
WGPU_BACKEND=metal cargo run        # explicit; the only real option on macOS
```

In fish, `RUST_LOG=... cargo run` works as written — fish supports the
`VAR=value cmd` prefix form. For a whole session, `set -x RUST_LOG wgpu_core=warn`.

To pass arguments to *your* program rather than to cargo, put them after `--`:

```bash
cargo run -- --agents 500000
```

Everything before `--` is cargo's; everything after is `std::env::args()`.

---

## 5. Tests

```bash
cargo test                      # everything
cargo test diffuse              # only tests whose name contains "diffuse"
cargo test -- --nocapture       # show println! from passing tests too
cargo test -- --test-threads=1  # run serially
cargo test --release            # when a test is too slow in debug
```

Cargo compiles tests from three places:

- **`#[cfg(test)] mod tests` inside `src/*.rs`** — unit tests, can see private
  items. Put anything testing internal helpers here.
- **`tests/*.rs`** — integration tests. Each file becomes its own binary and can
  only use your crate's public API. For a binary crate like this one, that means
  nothing is importable unless you also add a `src/lib.rs` — worth doing once
  the simulation logic is real, and the reason gpu-sim-course splits `common/`
  out as a library.
- **doc comments** — `cargo test` also runs ```` ``` ```` examples in `///`
  comments.

`cargo test` captures stdout from passing tests and shows it only for failures.
That is why a `println!` you added "to check" appears to do nothing;
`-- --nocapture` is the fix.

### Testing GPU code specifically

You do not need a window. Create a headless device — instance, adapter, device,
queue, no surface — and the whole compute pipeline works. Two things will bite:

1. **One shared device for the whole test binary.** `cargo test` runs tests on
   parallel threads. If every test creates its own `wgpu::Instance`, you get
   several live devices each with a thread parked in
   `poll(PollType::wait_indefinitely())`, and that combination deadlocks — the
   test binary hangs with no output, which looks exactly like an infinite loop
   in your kernel. Put the device behind a `static OnceLock<GpuContext>` and
   share it. wgpu handles are `Send + Sync` and are internally refcounted, so
   this is both safe and much faster than paying device creation per test.
2. **Readback is a two-buffer dance.** You cannot map the storage buffer your
   shader wrote to — `MAP_READ` may only combine with `COPY_DST`. Allocate a
   separate staging buffer, `copy_buffer_to_buffer` into it, `map_async`, then
   `device.poll(PollType::wait_indefinitely())?` to make the mapping actually
   resolve. Forgetting the poll gives you a callback that never fires.

   For textures, also remember the 256-byte `bytes_per_row` alignment — see
   [01-what-can-read-write-what.md](01-what-can-read-write-what.md#copies).

Note that **`cargo add` does not work offline**, and that includes
`[dev-dependencies]`. If you want an assertion helper like `approx`, add it
while you still have network. Plain `assert!((a - b).abs() < 1e-5)` needs
nothing.

---

## 6. Lint and format

```bash
cargo fmt              # rewrite in place
cargo fmt --check      # fail instead of rewriting
cargo clippy           # the lints that actually catch bugs
cargo clippy --fix     # auto-apply the mechanical ones
```

Both components are already installed, so both work offline.

---

## 7. Reading the output

- **Errors are ordered worst-first-ish, but fix the *first* one.** One missing
  type annotation cascades into a dozen downstream errors that vanish on their
  own.
- `cargo build 2>&1 | head -40` when the wall of text scrolls past.
- `cargo run --message-format=short` for one-line-per-error output, which is
  much easier to scan.
- **wgpu validation errors are runtime, not compile-time.** A bad bind group
  layout compiles fine and panics when you create the pipeline. The message
  quotes resource labels, which is why every descriptor in this project should
  get `label: Some("...")` — otherwise you get
  `<BindGroup-(0, 7, Metal)>` and no idea which one that is.
- **WGSL errors are also runtime.** `create_shader_module` parses the shader
  when you call it. naga's messages are good — they carry line and column — but
  they only appear when that line of Rust executes.

---

## 8. Build times, and why yours are short

`target/` is already warm: a full clean build of the 383-crate dependency tree
has been done once. So:

| Change | Rebuild cost |
|---|---|
| Edit a `.wgsl` file | none — shaders compile at runtime |
| Edit `src/*.rs` | seconds |
| Edit `Cargo.toml` profile section | full rebuild of everything |
| `cargo clean` | ~10 minutes, and it deletes the offline docs |

The third row is the trap. Touching `[profile.dev]` invalidates every compiled
artifact, dependencies included. Decide on profile settings once and leave them.

`cargo build --timings` writes an HTML flame chart to
`target/cargo-timings/` if you ever want to know where a build went.
