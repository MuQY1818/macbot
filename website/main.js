// macbot website: language toggle, synchronized terminal + scene packet, console carousel, reveal.
(() => {
  const ZH = {
    "nav.workflow": "流程", "nav.speed": "速度", "nav.console": "控制台", "nav.support": "支持", "nav.docs": "文档",
    "hero.title": "在 <span class=\"mark\">MacBook</span> 上跑 LeRobot。",
    "hero.lede": "在 Apple 芯片的 Mac 上录制示范，把 LeRobot 策略直接跑在真实机械臂上。",
    "hero.copy": "复制", "hero.quickstart": "快速开始", "hero.migrate": "从 Linux 迁移",
    "rail.init": "选择机械臂和相机", "rail.collect": "录制数据", "rail.export": "LeRobot v3.0 数据集",
    "rail.train": "在 GPU 机器上", "rail.run": "权重上机", "rail.replay": "画面、关节、动作块",
    "st.chunk": "/ 块", "st.mean": "平均",
    "wf.kicker": "流程", "wf.title": "从遥操作到策略上机。",
    "wf.sub": "所有数据都是 LeRobot 格式，数据集和权重在 Mac 和 GPU 机器之间拷来拷去就能用。",
    "wf.s1d": "选择机械臂和相机，检查已有的 LeRobot 校准。已有 LeRobot 命令？用 <code>--from-lerobot</code> 直接导入。",
    "wf.s2d": "边看相机画面边录制，每条都会自动检查。还没有机械臂？试试 <code>collect --fake</code>。",
    "wf.s3d": "导出 LeRobot v3.0 数据集。<code>health</code> 也能直接检查本地或 Hub 上已有的数据集。",
    "wf.s4d": "在任意 GPU 机器上用 LeRobot 自带的训练脚本训练，再把权重拷回来。",
    "wf.s5d": "加载本地或 Hub 上的权重，每次试验的画面和时序都会记录下来。",
    "wf.s6d": "把试验视频、关节曲线和动作块对齐到同一条时间线上。",
    "sp.kicker": "速度", "sp.title": "MacBook Pro 上每个动作块的耗时。",
    "sp.sub": "M5 Pro，48 GB。π0.5、SmolVLA 和 X-VLA 去噪 10 步，PyTorch 用 bf16，MLX 用 fp16；ACT 两边都用 fp32。",
    "sp.note": "动作块长度：π0.5 和 SmolVLA 为 50，X-VLA 为 30，ACT 为 100，都在录制的真实观测上测量。π0.5 是一个 SO-101 微调权重；SmolVLA（<code>lerobot/smolvla_base</code>）、X-VLA（SO-101 微调）和 ACT（ALOHA 方块转移）都用公开权重。RTC 即 Real-Time Chunking。π0.5 需要至少 32 GB 内存。",
    "sp.more": "方法与完整结果 →",
    "cs.kicker": "控制台", "cs.title": "每一步都有自己的浏览器控制台。",
    "cs.collect": "采集", "cs.run": "部署", "cs.replay": "回放",
    "mac.title": "macOS 上的几个坑。",
    "mac.c1": "sudo 下的驱动", "mac.c1d": "sudo 会丢掉库路径。macbot 自己加载 PCBUSB CAN 驱动，并把 root 录下的文件交还给你。",
    "mac.c2": "选对相机", "mac.c2d": "macOS 喜欢把附近的 iPhone 当成相机。macbot 按唯一 ID 选相机，从不按名称或序号。",
    "mac.c3": "推理加速", "mac.c3d": "π0.5、π0、SmolVLA、X-VLA 和 ACT 都有 MLX 后端，和 PyTorch 对照过误差才会被自动选用。",
    "mac.c4": "安全", "mac.c4d": "控制台启动时是暂停状态，限制关节速度和夹爪力度，相机卡住时自动暂停，电机断电前先把机械臂慢慢放下。",
    "sup.kicker": "支持", "sup.title": "测过什么。", "sup.robot": "机械臂", "sup.policy": "策略", "sup.status": "状态",
    "sup.hw": "真机", "sup.fake": "假硬件", "sup.sim": "MuJoCo 中的 SO-101 桌面场景", "sup.simb": "仿真",
    "sup.verified": "已验证", "sup.runs": "可运行（小权重）", "sup.others": "等", "sup.no": "未验证",
    "sup.robots": "机械臂支持", "sup.policies": "策略支持",
    "foot.text": "基于 LeRobot。π0.5 和数据增强方案来自 openpi，Real-Time Chunking 来自 Black 等人 2025。reBot 机械臂来自 Seeed Studio。Apache-2.0。"
  };
  const CAPS = {
    en: { collect: "Live cameras, joint state and a check of each recorded episode.",
          run: "Run a checkpoint on the arm and record each trial with cameras and timing.",
          replay: "Replay trial video, joint traces and action chunks on one timeline." },
    zh: { collect: "实时相机画面、关节状态，以及每条录制数据的检查。",
          run: "在机械臂上运行权重，记录每次试验的画面和时序。",
          replay: "同步回放试验视频、关节曲线和动作块。" }
  };

  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  // ---------- language ----------
  const nodes = $$("[data-i18n]");
  nodes.forEach(n => { n.dataset.en = n.innerHTML; });
  const links = $$("[data-href-zh]");
  links.forEach(a => { a.dataset.hrefEn = a.getAttribute("href"); });
  let lang = localStorage.getItem("macbot-lang") || ((navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en");
  function applyLang() {
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    nodes.forEach(n => { n.innerHTML = lang === "zh" && ZH[n.dataset.i18n] ? ZH[n.dataset.i18n] : n.dataset.en; });
    links.forEach(a => a.setAttribute("href", lang === "zh" ? a.dataset.hrefZh : a.dataset.hrefEn));
    $("#lang").textContent = lang === "zh" ? "EN" : "中文";
    showShot(shot, true);
  }
  $("#lang").addEventListener("click", () => { lang = lang === "zh" ? "en" : "zh"; localStorage.setItem("macbot-lang", lang); applyLang(); });

  // ---------- copy ----------
  $("#copy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText("pip install macbot"); } catch (e) { /* clipboard unavailable */ }
    const b = $("#copy"); const old = b.innerHTML; b.classList.add("done"); b.textContent = "✓";
    setTimeout(() => { b.classList.remove("done"); b.innerHTML = old; }, 1300);
  });

  // ---------- console carousel ----------
  const SHOTS = ["collect", "run", "replay"];
  const DUR = 7000;
  let shot = "collect", shotTimer = null, consoleOn = false;
  new IntersectionObserver(es => { consoleOn = es[0].isIntersecting; }).observe(document.getElementById("console"));
  function showShot(name, instant) {
    shot = name;
    const img = $("#shot");
    $$(".tab").forEach(t => { t.classList.remove("on"); void t.offsetWidth; t.classList.toggle("on", t.dataset.shot === name); });
    $("#shotcap").textContent = CAPS[lang][name];
    const src = `img/${name}_${lang}.jpg`;
    if (instant || reduce) { img.src = src; } else {
      img.classList.add("wipe");
      const pre = new Image(); pre.onload = () => { img.src = src; requestAnimationFrame(() => img.classList.remove("wipe")); }; pre.src = src;
    }
    clearTimeout(shotTimer);
    if (!reduce) shotTimer = setTimeout(function next() {
      if (!consoleOn || document.hidden) { shotTimer = setTimeout(next, 800); return; }
      showShot(SHOTS[(SHOTS.indexOf(name) + 1) % SHOTS.length]);
    }, DUR);
  }
  document.documentElement.style.setProperty("--dur", DUR / 1000 + "s");
  $$(".tab").forEach(t => t.addEventListener("click", () => showShot(t.dataset.shot)));

  applyLang();

  // ---------- hero: terminal + packet through the scene ----------
  // Scene coordinates are in the units of assets/scene.svg (viewBox -10 80 1345 345).
  const SX = x => (x + 10) / 1345 * 100, SY = y => (y - 80) / 345 * 100;
  const P = {
    collect: [[455, 222.5], [600, 222.5]],
    export: [[915, 222.5], [1118, 222.5]],
    back: [[1118, 292.5], [918, 292.5]],
    run: [[598, 292.5], [465, 292.5]]
  };
  const pk = $("#packet");
  const box = $("#scene");
  let bw = box.clientWidth, bh = box.clientHeight;
  new ResizeObserver(() => { bw = box.clientWidth; bh = box.clientHeight; }).observe(box);
  // Pause the hero loop when it is off screen or the tab is hidden.
  let heroOn = true;
  new IntersectionObserver(es => { heroOn = es[0].isIntersecting; }).observe($(".hero"));
  const idle = async () => { while (!heroOn || document.hidden) await sleep(400); };
  const place = (x, y) => { pk.style.transform = `translate(${SX(x) / 100 * bw - pk.offsetWidth / 2}px, ${SY(y) / 100 * bh - pk.offsetWidth / 2}px)`; };
  const term = $("#term");
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const STAGES = [
    { id: "init", zone: null, cmd: "macbot init --from-lerobot \"$LEROBOT_COMMAND\"" },
    { id: "collect", zone: "arms", cmd: "macbot collect", move: ["collect", "", "right"] },
    { id: "export", zone: "mac", cmd: "macbot export --out ./dataset", move: ["export", "", "right"] },
    { id: "train", zone: "gpu", note: "# lerobot-train on a GPU machine", hold: 1400 },
    { id: "run", zone: "mac", cmd: "macbot run --policy ./pretrained_model --backend mlx", move: ["back", "teal", "left"], move2: ["run", "", "left"] },
    { id: "replay", zone: null, cmd: "macbot replay" }
  ];
  let lines = [];
  const render = (typing) => { term.innerHTML = lines.join("") + (typing || "") + '<span class="caret"></span>'; };
  const cmdLine = t => `<span class="p">$</span> ${esc(t).replace(/^macbot/, '<span class="k">macbot</span>')}\n`;

  function setStage(st) {
    $("#stage-name").textContent = st.id;
    $$("#rail li").forEach(li => li.classList.toggle("on", li.dataset.stage === st.id));
    $$(".zone").forEach(z => { z.classList.toggle("on", z.dataset.zone === st.zone); z.classList.toggle("gpuz", z.dataset.zone === "gpu"); });
  }
  // Runs on the compositor: one transform/opacity animation per hop, no per-frame script.
  const at = (x, y) => `translate(${SX(x) / 100 * bw - pk.offsetWidth / 2}px, ${SY(y) / 100 * bh - pk.offsetWidth / 2}px)`;
  function fly([seg, color], ms) {
    const [[x0, y0], [x1, y1]] = P[seg];
    pk.className = "packet " + color;
    const a = pk.animate([
      { transform: at(x0, y0), opacity: 0 },
      { transform: at(x0 + (x1 - x0) * .08, y0 + (y1 - y0) * .08), opacity: 1, offset: .08 },
      { transform: at(x0 + (x1 - x0) * .92, y0 + (y1 - y0) * .92), opacity: 1, offset: .92 },
      { transform: at(x1, y1), opacity: 0 }
    ], { duration: ms, easing: "cubic-bezier(.45, 0, .55, 1)", fill: "forwards" });
    return a.finished;
  }
  async function typeCmd(t) {
    await idle();
    for (let i = 1; i <= t.length; i++) { render(`<span class="p">$</span> ${esc(t.slice(0, i))}`); await sleep(18 + Math.random() * 34); }
    lines.push(cmdLine(t)); render();
  }
  let manual = null;
  async function loop() {
    if (reduce) {
      lines = ["<span class=\"c\"># Apple silicon, Python 3.12+</span>\n", cmdLine("pip install macbot"), ...STAGES.map(s => s.cmd ? cmdLine(s.cmd) : `<span class="c">${esc(s.note)}</span>\n`)];
      render(); return;
    }
    for (;;) {
      lines = ["<span class=\"c\"># Apple silicon, Python 3.12+</span>\n"];
      render();
      await typeCmd("pip install macbot");
      await sleep(300);
      for (const st of STAGES) {
        setStage(st);
        if (st.cmd) await typeCmd(st.cmd);
        if (st.note) { lines.push(`<span class="c">${esc(st.note)}</span>\n`); render(); }
        await idle();
        if (st.move) await fly(st.move, 1300);
        if (st.move2) { await sleep(150); await fly(st.move2, 1100); }
        await sleep(st.hold || 650);
        while (manual) await sleep(200);
      }
      await sleep(2600);
    }
  }
  $$("#rail li").forEach(li => {
    li.addEventListener("mouseenter", () => { manual = li.dataset.stage; setStage(STAGES.find(s => s.id === manual)); });
    li.addEventListener("mouseleave", () => { manual = null; });
  });
  loop();

  // ---------- reveal ----------
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add("in");
    io.unobserve(e.target);
  }), { threshold: 0.2 });
  $$(".reveal").forEach(el => io.observe(el));
})();
