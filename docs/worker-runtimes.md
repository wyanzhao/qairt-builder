# Worker runtimes and harness upgrades

The CLI is the primary automation boundary. `worker.backend = "auto"` is
deterministic:

| Host | Runtime | Worker |
|---|---|---|
| macOS | Apple `container` | Ubuntu 22.04, Python 3.10, `linux/amd64` under Rosetta on Apple Silicon |
| Linux | Docker | Ubuntu 22.04, Python 3.10, `linux/amd64` |

`worker.backend = "native"` remains an explicit opt-in. `doctor` treats the
host ABI as critical in that mode and requires the harness Ubuntu, x86_64, and
Python versions. The project never installs Docker, Apple `container`, a Linux
kernel, Rosetta, or privileged DNS configuration.

## macOS initialization

The supported Apple CLI version is declared in
`harness/constraints.json`. After installing that version, start its services
yourself:

```bash
container --version
container system start
qairt-agent image build --root .
qairt-agent image smoke --root .
qairt-agent doctor --root .
```

Both `container --version` and the running `container-apiserver` version must
exactly match the harness. This catches a stale service left running after a
CLI upgrade.

The image and SDK are not baked together. The worker image contains Ubuntu,
Python, ADB, and pinned Python dependencies; QAIRT is mounted read-only at
`/opt/qairt`. Project and model inputs are read-only, while state, jobs,
artifacts, and cache have explicit writable mounts.

An ADB server bound to host loopback is not directly reachable from Apple's
VM. Apple `container` 1.0 requires a privileged localhost DNS bridge. Configure
it explicitly once, and repeat it after a host restart if the packet-filter
rule was removed:

```bash
sudo container system dns create host.container.internal \
  --localhost 203.0.113.113
```

The agent never runs this command. When
`QAIRT_AGENT_ADB_SERVER=localhost:5037`, worker launch first verifies the domain
with `container system dns list --format json`, then passes
`host.container.internal:5037` into the worker while retaining
`localhost:5037` as the canonical device-lease identity.

Apple's 1.0 CLI documents `--platform linux/amd64`, `--rosetta`,
`--user`, `--workdir`, `--env`, and bind mounts. It does not expose a
Docker-equivalent `--network none`; image smoke uses `--no-dns` and does not
claim hard egress isolation.

## Linux initialization

Linux uses Docker even when the host happens to match the worker ABI:

```bash
docker version
qairt-agent image build --root .
qairt-agent image smoke --root .
qairt-agent doctor --root .
```

Set `worker.backend = "native"` only when native execution is intentional.
Both Docker client and daemon must satisfy the harness minimum version.
Docker maps a loopback ADB server to `host.docker.internal` and adds Docker's
host-gateway mapping. Both runtime aliases canonicalize back to the same
server+serial device lease.

## Updating the harness

All compatibility pins are reviewed together in
`harness/constraints.json`:

- QAIRT version and build ID
- Ubuntu, Python, and OCI platform
- worker image tag and Dockerfile
- pinned Python dependency file
- Torch version and wheel index
- Apple `container` exact version and Docker minimum version
- target chipset, DSP architecture, and SoC model

The Dockerfile consumes these values as build arguments; it does not own a
second copy of the pins. The selected project constraints file must remain
inside the project/build context. Image smoke validates the exact SDK
version/build, Python ABI, Torch pin, and every exact `name==version` entry in
the selected dependency lock. `qairt-agent init` stages the exact running
agent's Python sources as a deterministic, Python-ABI-neutral zip; the image
imports that archive after installing only the reviewed dependency lock. It
does not build or install an arbitrary package from the model project's root.
For an upgrade:

1. Edit `harness/constraints.json`.
2. Add or update the referenced requirements file.
3. Run the focused harness/runtime tests.
4. Build the image with `qairt-agent image build --root .`.
5. Run `image smoke`, `doctor`, then a real device acceptance workflow.

`qairt-agent init` copies the distribution's constraints, Dockerfile, and
matching dependency lock into a new project, generates the worker source
archive, and installs a final managed `.dockerignore` block. This is identical
for an editable checkout and an installed wheel. The generated
`qairt-agent.toml` references `harness/constraints.json` instead of copying its
version values. Legacy `[docker]` values are treated only as explicit
overrides. A changed `worker.dependencies_file` remains fail-closed until that
reviewed lock is present; it never falls back to the previous bundled lock.
