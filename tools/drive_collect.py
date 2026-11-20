"""Drive `macbot collect --fake` in-process: record one episode, save it, quit."""
import sys, threading, time
from macbot.config import load
import macbot.collect.server as S
cfg = load(sys.argv[1])
holder = {}
class St(S.State):
    def __init__(s, *a, **k):
        super().__init__(*a, **k); holder["st"] = s
S.State = St
th = threading.Thread(target=S.run, args=(cfg,), kwargs=dict(fake=True, port=int(sys.argv[2])), daemon=True)
th.start()
t = time.time()
while not (holder.get("st") and holder["st"].ready) and time.time() - t < 60:
    time.sleep(0.2)
st = holder["st"]
print("joints", st.joints, "limits", st.limits)
st.command({"action": "start"}); time.sleep(3.5)
st.command({"action": "save"})
while st.saving or st.phase != "idle":
    time.sleep(0.2)
time.sleep(0.5)
print("msg", st.msg["en"])
st.command({"action": "quit"}); th.join(15)
print("episodes", [p.name for p in st.files()])
