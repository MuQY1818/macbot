# Versioning and releases

[简体中文](releasing.zh.md)

macbot follows [Semantic Versioning](https://semver.org/) with the usual rule for 0.x projects. A version
is `MAJOR.MINOR.PATCH`.

## What counts as the public interface

A change is "breaking" if it can make a working setup stop working or behave differently without the user
changing anything. The public interface is:

- CLI commands, their flags and their defaults (`macbot collect`, `run`, `export`, `bench`, ...);
- YAML configuration keys and their meaning (`macbot/default.yaml`);
- file formats on disk: raw episodes, rollouts, `gripper.json`, and the layout of exported LeRobot datasets;
- the Python extension points: the `Robot` interface (`macbot/robot/base.py`), camera types, executors;
- default runtime behaviour that changes what the robot does: default backend, execution scheme, speed and
  joint limits, safety stops;
- supported Python and LeRobot versions.

Internal modules, tool scripts under `tools/`, log text and web page layout are not part of it.

## While macbot is 0.x (now)

| Bump | When | Examples |
|---|---|---|
| `0.MINOR.0` | Something in the public interface breaks, or a default that changes robot behaviour changes. | A config key is renamed or removed; the episode format changes; `--mode auto` picks a different scheme; LeRobot requirement moves to a new minor version. |
| `0.x.PATCH` | Everything that is backward compatible: bug fixes, new opt-in features, new robots or policies, new flags with a safe default, speed-ups that keep outputs, docs. | The MLX RTC path; a new `--mlx-time-grid` flag; support for another LeRobot arm; a crash fix. |

New features therefore usually ship as patch releases while macbot is 0.x. A minor bump is a signal to read
the changelog before upgrading. Every breaking change is listed under **Changed** or **Removed** with what to do.

## From 1.0.0 on

1.0.0 means the interface above is stable: at least two arm families verified on hardware, the episode and
rollout formats frozen, and the config schema documented. After that, standard SemVer applies:
`MAJOR` for breaking changes, `MINOR` for new backward-compatible features, `PATCH` for fixes only.

## How a release happens

1. Changes land on `main` through pull requests with green CI, and each one adds a line under
   `## [Unreleased]` in both changelogs.
2. Releases are cut when there is something worth installing, not after every merge. A safety or crash fix is
   released promptly as a patch.
3. Work that should be tried before it is relied on (a new backend, a new default) can go out as a
   pre-release first, for example `0.4.0rc1`. pip only installs it when asked (`pip install --pre macbot`).
4. To release: pick the number by the tables above, move `Unreleased` into a dated section in
   `CHANGELOG.md` and `CHANGELOG_zh.md`, set the version in `pyproject.toml` and `macbot/__init__.py`, merge,
   then publish a GitHub release `vX.Y.Z` from `main`. The release workflow uploads to PyPI through Trusted
   Publishing.
5. A broken release is fixed forward with a new patch. A release that is dangerous to install is also yanked
   on PyPI.
