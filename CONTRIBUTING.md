# Contributing to macbot

[简体中文](CONTRIBUTING_zh.md)

## Development setup

Use Python 3.12 or 3.13 in a dedicated environment. From the checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pyflakes macbot tests tools
pytest -q
```

Use Ruff for linting and aim for 120 columns. Keep style fixes local to your change.
To run checks before each commit, use `pre-commit install`.

CI runs the fast hardware-free tests on macOS and Ubuntu. To reproduce its minimal environment,
in a fresh virtual environment with pip 25.1 or newer, run:

```bash
python -m pip install --no-deps -e .
python -m pip install --group lint --group test
pytest -q -rs
```

Tests that require LeRobot use `pytest.importorskip`. Test web servers in-process:
drive their state or handlers directly, without loopback HTTP. See [the collection driver](tools/drive_collect.py).

## Try changes without hardware

```bash
macbot --config configs/so101.yaml collect --fake
macbot --config configs/so101.yaml health --data ~/macbot-data/raw/pick_cube_fake
```

In the collection page, press **S** to start and **Enter** to save. For a simulator check, install
`python -m pip install -e ".[dev,sim]"` and run `macbot --config configs/so101_sim.yaml collect`
without `--fake`. The pinned scene downloads and executes Python on first use; see the
[simulation setup](docs/setup.md#local-mujoco-simulator) and [workflow](docs/guide.md#simulation).
For policy inference, use a checkpoint matching the joint layout and cameras; `run --fake` still runs the real policy.

## Extending macbot

- **Robot:** Start with [robot support](docs/robots.md). Prefer a compatible LeRobot config; add a
  `Robot` implementation only if the device needs different behavior. Test the joint order, action
  units, calibration guard and shutdown behavior without hardware.
- **Execution scheme:** Implement the `Executor` interface in
  [`macbot/policy/executors.py`](macbot/policy/executors.py). To change how an asynchronous chunk
  replaces the active one, override `AsyncChunk.accept()`. Wire the mode into `macbot/cli.py`
  and the runner, and test the timing and pause/reset behavior with `FakePolicy`.
  See [architecture](docs/architecture.md).

## Commits and pull requests

Keep commits small and focused, with plain English subjects. Run the checks above before each
commit. Open a pull request with the behavior change, how to verify it, and the results. Update
the English and Simplified Chinese docs together and add an entry under **Unreleased** for
user-visible changes. How versions are numbered and released is described in [docs/releasing.md](docs/releasing.md).

Hardware test reports are welcome. Include the arm and camera setup, macOS and chip, what worked
or failed, and `macbot bench` numbers when available. Use the hardware report issue form and
remove personal data from any logs or screenshots.
