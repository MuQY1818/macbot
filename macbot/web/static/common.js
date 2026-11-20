// Shared helpers for the macbot pages.
const $ = id => document.getElementById(id);
const esc = x => String(x).replace(/[&<>"]/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[m]));
// ---------- language (EN / 中文), remembered per browser
const I18N = {
  en: {
    quit: "Quit & home", collect: "collect", run: "run", replay: "replay", instruction: "instruction",
    quit_sim: "Quit", reset_scene: "Reset scene", render_hold: "still frame",
    record: "Record", save: "Save", discard: "Discard", set_ins: "Set instruction", progress: "progress",
    joints: "joints", joints_hint: "bar = command · line = measured", episodes: "episodes",
    th_len: "length", th_frames: "frames", th_ins: "instruction", th_saved: "saved", th_check: "check",
    no_eps: "no episodes yet", play: "Play", pause: "Pause", back_live: "Back to live", del: "Delete",
    rec: "rec", ready: "ready", simulated: "simulated", not_ready: "not ready", connecting: "connecting",
    disconnected: "disconnected", saving: "saving…", live: "live", no_signal: "no signal", frames: "frames",
    saved_n: n => `${n} saved`, eps_of: (a, b, f) => `${a} / ${b} episodes · ${f} fps`, data: "data",
    crop: "crop", full: "full frame", ok: "ok", warn: "warn", fail: "fail",
    confirm_del: i => `Delete episode ${i}? The file moves to deleted/.`,
    confirm_quit: "Quit? The follower returns to zero slowly, then disables.",
    legend: "solid = measured · dashed = command", tagline: "real-robot learning on a MacBook",
    replay_title: (i, s) => `replay · episode ${i} · ${s} s`,
    start: "Start", hold: "Pause", home: "Home", succ: "Success", failb: "Failure", drop: "Discard run",
    running: "running", paused: "paused", homing: "homing", status: "status", chunks: "chunks",
    infer: "inference", run_time: "this run", next: "next", executor: "executor", speed: "speed",
    policy: "policy", trials: "trials", no_trials: "no trials yet", success: "success", failure: "failure",
    rollouts: "rollouts", replay_btn: "Replay", result: "result", length: "length", no_rollouts: "no rollouts in this folder",
    chunk_marks: "ticks = new chunk", recent: "recent 15 s",
    saved_trials: "Saved trials", trial_heading: n => `Trial ${n}`, live_joints: "Live joints",
    show_all: n => `Show all (${n})`, show_less: "Show recent",
    pause_first: "Pause first to replay a saved trial.", replay_busy: "Replay is busy. Try again.",
    replay_failed: "Could not load replay.", replay_missing: "This trial or frame is no longer available.",
    replay_loading: "Loading trial…", trial_note: "Note", review_saved: "Review saved.",
    review_failed: "Could not save this review.", playback_speed: "Playback speed",
    playback_position: "Playback position", chunk_at: n => `Chunk ${n}`, median_infer: "median inference",
    takeover: "Take over", return_policy: "Return to policy", following: "Following leader", aligning: "Aligning arms",
    leader_delta: "Leader − follower", align_target: n => `Within ±${n}°`, leader_missing: "Leader reading unavailable",
    mark_hand: "Mark hand in scene", hand_marked: n => `${n} hand marks`, hand_in_scene: "Hand in scene",
    grasp_miss: "Grasp miss", grasp_misses: "Misses", grasp_unknown: "Unknown", grasp_uncertain: n => `${n} inconclusive closes`,
    exclude_trial: "Exclude trial", exclude_reason: "Reason for exclusion", excluded: "Excluded",
    reviewed_rate: (ok, n) => `${ok} / ${n} successful`,
  },
  zh: {
    quit: "结束并回零", collect: "采集", run: "跑策略", replay: "回放", instruction: "语言指令",
    record: "开始录制", save: "保存", discard: "丢弃", set_ins: "保存指令", progress: "进度",
    joints: "关节", joints_hint: "色块 = 指令 · 竖线 = 实测", episodes: "已采数据",
    th_len: "时长", th_frames: "帧数", th_ins: "指令", th_saved: "保存时间", th_check: "检查",
    no_eps: "还没有保存的数据", play: "播放", pause: "暂停", back_live: "回到实时画面", del: "删除",
    rec: "录制中", ready: "待机", simulated: "模拟", not_ready: "未就绪", connecting: "连接中",
    quit_sim: "退出", reset_scene: "重置场景", render_hold: "静止画面",
    disconnected: "已断开", saving: "保存中…", live: "实时", no_signal: "没有画面", frames: "帧",
    saved_n: n => `共 ${n} 条`, eps_of: (a, b, f) => `${a} / ${b} 条 · ${f} fps`, data: "数据",
    crop: "裁剪", full: "全幅", ok: "通过", warn: "注意", fail: "不通过",
    confirm_del: i => `删除第 ${i} 条？文件会移到 deleted/。`,
    confirm_quit: "结束程序？从臂会先慢速回零，再失能。",
    legend: "实线 = 实测 · 虚线 = 指令", tagline: "一台 MacBook 搞定真机学习",
    replay_title: (i, s) => `回放 · 第 ${i} 条 · ${s} 秒`,
    start: "开始", hold: "暂停", home: "回起始姿态", succ: "存为成功", failb: "存为失败", drop: "丢弃这次",
    running: "运行中", paused: "已暂停", homing: "回起始姿态", status: "运行信息", chunks: "推理块数",
    infer: "推理耗时", run_time: "本次已运行", next: "下一条", executor: "执行方式", speed: "速度",
    policy: "策略", trials: "已保存的试验", no_trials: "还没有试验", success: "成功", failure: "失败",
    rollouts: "试验记录", replay_btn: "回放", result: "结果", length: "时长", no_rollouts: "这个文件夹里没有试验记录",
    chunk_marks: "刻度 = 换块", recent: "最近 15 秒",
    saved_trials: "已保存的试验", trial_heading: n => `试验 ${n}`, live_joints: "实时关节",
    show_all: n => `显示全部（${n}）`, show_less: "只显示最近试验",
    pause_first: "请先暂停，再回放已保存的试验。", replay_busy: "回放繁忙，请重试。",
    replay_failed: "无法加载回放。", replay_missing: "这条试验或这一帧已不可用。",
    replay_loading: "正在加载试验…", trial_note: "备注", review_saved: "已保存试验评审。",
    review_failed: "无法保存试验评审。", playback_speed: "回放速度",
    playback_position: "回放位置", chunk_at: n => `第 ${n} 块`, median_infer: "推理耗时中位数",
    takeover: "手动接管", return_policy: "交还策略", following: "跟随领臂", aligning: "等待对齐",
    leader_delta: "领臂与从臂的角度差", align_target: n => `允许误差 ±${n}°`, leader_missing: "没有有效的领臂读数",
    mark_hand: "标记手进入画面", hand_marked: n => `${n} 处手动干预标记`, hand_in_scene: "手进入画面",
    grasp_miss: "空抓", grasp_misses: "空抓数", grasp_unknown: "未知", grasp_uncertain: n => `${n} 次闭合无法判定`,
    exclude_trial: "排除此试验", exclude_reason: "排除原因", excluded: "已排除",
    reviewed_rate: (ok, n) => `${n} 次试验中 ${ok} 次成功`,
  },
};
let LANG = (() => { try { return localStorage.getItem("macbot.lang") || ((navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en"); } catch (e) { return "en"; } })();
const t = (k, ...a) => { const v = (I18N[LANG] || I18N.en)[k] ?? I18N.en[k] ?? k; return typeof v === "function" ? v(...a) : v; };
const L = m => (m && typeof m === "object") ? (m[LANG] || m.en || "") : (m ?? "");     // server messages {en, zh}
const CAM_LABEL = { zh: { top: "俯视", wrist: "腕部", front: "前视", side: "侧视" } };
const camLabel = c => (CAM_LABEL[LANG] || {})[c] || c;

function applyI18n() {
  document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-label]").forEach(el => el.setAttribute("aria-label", t(el.dataset.label)));
  const b = document.getElementById("lang");
  if (b) b.textContent = LANG === "zh" ? "EN" : "中文";
}
function setLang(l) {
  LANG = l;
  try { localStorage.setItem("macbot.lang", l); } catch (e) { /* private mode */ }
  applyI18n();
  if (typeof window.onLang === "function") window.onLang();
}
document.addEventListener("DOMContentLoaded", () => {
  applyI18n();
  const b = document.getElementById("lang");
  if (b) b.onclick = () => setLang(LANG === "zh" ? "en" : "zh");
});

const JOINT_LABELS = {
  en: { shoulder_pan: "pan", shoulder_lift: "lift", elbow_flex: "elbow", wrist_flex: "w.flex", wrist_yaw: "w.yaw", wrist_roll: "w.roll", gripper: "gripper" },
  zh: { shoulder_pan: "肩转", shoulder_lift: "肩抬", elbow_flex: "肘", wrist_flex: "腕俯仰", wrist_yaw: "腕偏航", wrist_roll: "腕滚转", gripper: "夹爪" },
};
const jointLabel = n => {
  const labels = JOINT_LABELS[LANG] || JOINT_LABELS.en;
  return Object.hasOwn(labels, n) ? labels[n] : n;
};
const JOINT_COLORS = ["#F2701A", "#1BA9A0", "#6E5BE8", "#D93C3C", "#D9A21B", "#1F9D63", "#8A8597"];
const jointColor = j => JOINT_COLORS[j] ?? `hsl(${(j * 137.5) % 360} 65% 48%)`;
const fmtTime = t => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;

// Status polls carry the console process id. A page left open across sessions reloads when a new one starts,
// so `macbot run` reuses this tab instead of opening another.
let BOOT = null;
async function pollStatus(url = "/api/status") {
  const r = await fetch(url);
  const boot = r.headers.get("X-Macbot-Boot");
  if (boot && BOOT && boot !== BOOT) { location.reload(); return new Promise(() => {}); }
  BOOT = boot || BOOT;
  return r.json();
}

async function post(url, body) {
  const r = await fetch(url, { method: "POST", body: JSON.stringify(body) });
  return r.json();
}

// Joint meters: thick bar = command, thin line = measured, scaled to the joint limits.
// Camera views share one row at equal height: each view's width follows its own aspect ratio,
// so a square top view and a 16:9 wrist view line up without cropping or empty space.
function layoutViews(el, names, sizeOf, prefix = "") {
  el.innerHTML = names.map(c => {
    const [w, h] = sizeOf(c) || [4, 3];
    return `<div class="view" style="--ar:${(w / h).toFixed(4)}"><img id="${prefix}cam-${esc(c)}" alt="${esc(c)}">` +
      `<span class="tag" id="${prefix}tag-${esc(c)}">${esc(camLabel(c))}</span><span class="age" id="${prefix}age-${esc(c)}"></span></div>`;
  }).join("");
}

function renderJoints(el, names, q, cmd, limits) {
  el.innerHTML = names.map((n, i) => {
    const [lo, hi] = limits && Object.hasOwn(limits, n) ? limits[n] : [-180, 180];
    const p = v => Math.max(0, Math.min(100, 100 * (v - lo) / (hi - lo)));
    return `<div class="joint"><span class="name">${esc(jointLabel(n))}</span><div class="meter">` +
      `<span class="cmd" style="left:${p(cmd[i])}%"></span><span class="meas" style="left:${p(q[i])}%"></span></div>` +
      `<span class="val num">${q[i].toFixed(1)}°</span></div>`;
  }).join("");
}

// Segmented progress bar (25 cells).
function renderHP(el, frac) {
  const on = Math.round(25 * Math.max(0, Math.min(1, frac)));
  el.innerHTML = Array.from({ length: 25 }, (_, i) => `<i class="${i < on ? "on" : ""}"></i>`).join("");
}

// Joint traces for one episode: solid = measured, dashed = command, each joint scaled on its own.
function drawTraces(svg, legend, qpos, action, names, extra = "", fractions = null, background = "") {
  const n = qpos.length, W = 1000, H = 190, x = i => (W * (fractions?.[i] ?? i / Math.max(n - 1, 1))).toFixed(1);
  let s = background;
  for (let j = 0; j < names.length; j++) {
    const v = qpos.map(r => r[j]), a = action.map(r => r[j]);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { lo = Math.min(lo, v[i], a[i]); hi = Math.max(hi, v[i], a[i]); }
    const span = Math.max(hi - lo, 1), color = jointColor(j);
    const y = val => (H - 10 - (H - 20) * (val - lo) / span).toFixed(1);
    s += `<polyline points="${a.map((val, i) => `${x(i)},${y(val)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="1.2" stroke-dasharray="5 4" opacity=".55" vector-effect="non-scaling-stroke"/>`;
    s += `<polyline points="${v.map((val, i) => `${x(i)},${y(val)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  }
  s += extra + `<line data-cursor x1="0" x2="0" y1="0" y2="${H}" stroke="currentColor" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  svg.innerHTML = s;
  legend.innerHTML = names.map((nm, j) => `<span style="--c:${jointColor(j)}">${esc(jointLabel(nm))}</span>`).join("") +
    `<span style="--c:transparent">${t("legend")}</span>`;
}

function moveCursor(svg, k, n, fraction = k / Math.max(n - 1, 1)) {
  const c = svg.querySelector("[data-cursor]"), cx = 1000 * fraction;
  if (c) { c.setAttribute("x1", cx); c.setAttribute("x2", cx); }
}

function camTag(el, label, age, stall) {
  const dead = age === null || age === undefined || age > stall;
  el.textContent = `${camLabel(label)} · ${t(dead ? "no_signal" : "live")}`;
  el.classList.toggle("bad", dead);
}
