"""Record and save one episode in-process, without making HTTP requests."""
import argparse
import threading
import time
from concurrent.futures import Future
from unittest.mock import patch

from macbot.collect import server
from macbot.config import load


def wait_for(test, done, timeout):
    deadline = time.monotonic() + timeout
    while not test():
        if done.done():
            done.result()
            raise RuntimeError("collect exited before reaching the requested state")
        if time.monotonic() >= deadline:
            raise TimeoutError("collect did not reach the requested state")
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("port", type=int)
    parser.add_argument("--sim", action="store_true", help="use a configured simulator instead of --fake")
    parser.add_argument("--out")
    args = parser.parse_args()
    cfg = load(args.config)
    if args.sim and cfg["robot"].get("type") != "sim":
        parser.error("--sim requires robot.type: sim")
    created, done = Future(), Future()

    class CapturedState(server.State):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created.set_result(self)

    def collect():
        try:
            server.run(cfg, fake=not args.sim, port=args.port, out=args.out)
        except BaseException as exc:
            done.set_exception(exc)
        else:
            done.set_result(None)

    with patch.object(server, "State", CapturedState):
        worker = threading.Thread(target=collect, daemon=True)
        worker.start()
        wait_for(created.done, done, 60)
        st = created.result()
        try:
            wait_for(lambda: st.ready, done, 60)
            before = set(st.files())
            print("joints", st.joints, "limits", st.limits)
            st.command({"action": "start"})
            wait_for(lambda: st.phase == "recording", done, 5)
            time.sleep(3.5)
            st.command({"action": "save"})
            wait_for(lambda: not st.saving and st.phase == "idle", done, 15)
            saved = sorted(set(st.files()) - before)
            print("msg", st.msg["en"])
            if not saved:
                raise RuntimeError("collect did not save a new episode")
            print("episodes", [p.name for p in saved])
        finally:
            st.command({"action": "quit"})
            done.result(timeout=20)
            worker.join()


if __name__ == "__main__":
    main()
