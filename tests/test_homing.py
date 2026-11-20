from macbot.config import load
from macbot.policy.runner import homing_steps


def test_homing_moves_all_arm_joints_together_at_home_vel():
    cmd, home, gi = [40.0, -20.0, 5.0, 0.0, -60.0], [0.0, 0.0, 0.0, 0.0, 0.0], 4
    steps = homing_steps(cmd, home, gi, 0.5)
    assert steps[0] == 0.5                       # farthest arm joint at the limit
    assert abs(steps[1] - 0.25) < 1e-9 and abs(steps[2] - 0.0625) < 1e-9 and steps[3] == 0.0
    # Simulate: every arm joint reaches home on the same cycle.
    cycles = []
    for j in range(4):
        pos, n = cmd[j], 0
        while abs(pos - home[j]) > 1e-9:
            pos -= max(-steps[j], min(steps[j], pos - home[j])) if steps[j] else pos
            n += 1
        cycles.append(n)
    assert len({c for c in cycles if c}) == 1


def test_homing_is_slow_by_default():
    run = load()["run"]
    assert run["home_vel"] <= 30 < run["max_vel"] * run["speed"]
