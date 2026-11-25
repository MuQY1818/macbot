# macbot documentation

[简体中文](index.zh.md)

macbot runs LeRobot on an Apple silicon Mac. It sets up the arms and cameras, records demonstrations, exports
them as LeRobot datasets, and runs the trained policy on the robot, with MLX acceleration where it is faster.

<div class="grid cards" markdown>

-   **New to macbot**

    ---

    From an assembled arm pair to your first saved demonstration.

    [Quickstart](quickstart.md)

-   **Coming from LeRobot on Linux**

    ---

    Reuse your calibration, `lerobot-record` command, datasets and checkpoints.

    [Migrate from Linux](migrate.md)

-   **Make a policy faster**

    ---

    Which policies have an MLX backend, how much faster they run, and how to turn it on.

    [Acceleration](acceleration.md)

-   **No hardware yet**

    ---

    Try the consoles, keyboard teleoperation and evaluation in a simulated scene.

    [Simulation](simulation.md)

</div>

## The workflow

Every step reads and writes standard LeRobot formats, so you can leave the workflow at any point.

| Step | Command | Details |
|---|---|---|
| Set up arms and cameras | `macbot init` | [Quickstart](quickstart.md) |
| Record demonstrations | `macbot collect` | [Collect](guide.md#1-collect) |
| Check and export | `macbot check`, `macbot export` | [Check](guide.md#2-check-the-data), [Export](guide.md#3-export) |
| Train on a GPU machine | LeRobot training | [Train](guide.md#4-train) |
| Run the policy | `macbot run` | [Run a policy](guide.md#5-run-a-policy) |
| Review trials | `macbot replay` | [Review](guide.md#6-review) |

Supported arms are listed under [Robots](robots.md) and supported policies under [Policies](policies.md).
