Object.assign(I18N.en, {
  setup: "Setup", setup_title: "Set up your robot", setup_rescan: "Rescan devices", setup_back: "Back",
  setup_next: "Continue", setup_save: "Save and run doctor", setup_dark: "Dark mode", setup_light: "Light mode",
  setup_arms: "Arms", setup_cameras: "Cameras", setup_calibrate: "Calibration", setup_review: "Review",
  setup_model: "Arm model", setup_choose: "Choose…", setup_leader: "Leader", setup_follower: "Follower",
  setup_noSerial: "no USB serial", setup_unknown: "Unknown adapter", setup_noArms: "No USB arm adapters found.",
  setup_confirm: "Choose the model printed on your kit. USB identifies the adapter, not the arm.",
  setup_move: "Identify by movement", setup_moveHint: "With torque off, click and gently move only the leader for three seconds.",
  setup_moveTwo: "Select the two serial adapters to sample.", setup_moveFail: "No clear result. Assign the arms manually.",
  setup_moveOk: "Leader identified. Confirm the two roles before continuing.",
  setup_moving: "Move the leader now…", setup_skip: "Do not use", setup_role: "Recording role",
  setup_noCameras: "No cameras found. Connect a camera and rescan.",
  setup_cameraHint: "Match each preview to a recording role. iPhone and Continuity Camera are excluded.",
  setup_keepFraming: "Keep framing from the existing config",
  setup_newFraming: "Changing framing can affect a trained policy.",
  setup_unmatched: "Original cameras not matched",
  setup_savedSize: "Saved",
  setup_fullFrame: "Full frame",
  setup_preview: "Camera preview", setup_previewWait: "Waiting for preview…", setup_previewError: "Preview unavailable",
  setup_calHint: "For each missing file, run the command in a separate Terminal and follow LeRobot’s prompts. Move the joints by hand.",
  setup_calExtra: "If the calibration command cannot import its dependencies, install lerobot[core-scripts]==0.7.0.",
  setup_checkAgain: "Check again", setup_copy: "Copy command", setup_copied: "Command copied.",
  setup_missing: "Calibration needed", setup_found: "File found",
  setup_align: "Before starting collection, support the follower, clear the workspace and match the two arm poses.",
  setup_replace: "Replace the existing configuration", setup_collect: "Start collection",
  setup_open: "Open collection console", setup_starting: "Collection is starting. Open the console once it is ready.",
  setup_loading: "Detecting devices…", setup_working: "Working…", setup_connect: "Connect the arms and cameras, then rescan.",
});
Object.assign(I18N.zh, {
  setup: "设置", setup_title: "设置机器人", setup_rescan: "重新扫描", setup_back: "上一步",
  setup_next: "继续", setup_save: "保存并运行检查", setup_dark: "深色模式", setup_light: "浅色模式",
  setup_arms: "机械臂", setup_cameras: "相机", setup_calibrate: "校准", setup_review: "确认",
  setup_model: "机械臂型号", setup_choose: "请选择…", setup_leader: "主臂", setup_follower: "从臂",
  setup_noSerial: "无 USB 序列号", setup_unknown: "未知适配器", setup_noArms: "未找到机械臂的 USB 适配器。",
  setup_confirm: "请选择套件上标注的型号。USB 只能识别适配器。",
  setup_move: "通过动作识别", setup_moveHint: "点击后，用手轻轻移动未使能的主臂约三秒，保持从臂不动。",
  setup_moveTwo: "请先选中要读取的两个串口适配器。", setup_moveFail: "未能区分两臂，请手动分配。",
  setup_moveOk: "已识别主臂，请确认两臂的角色后继续。",
  setup_moving: "现在移动主臂…", setup_skip: "不使用", setup_role: "录制角色",
  setup_noCameras: "未找到相机。请接入相机并重新扫描。",
  setup_cameraHint: "根据预览选择相机角色。已排除 iPhone 和连续互通相机。",
  setup_keepFraming: "保留现有配置的取景",
  setup_newFraming: "更改取景可能影响已训练的策略。",
  setup_unmatched: "未匹配的原相机",
  setup_savedSize: "保存尺寸",
  setup_fullFrame: "全幅",
  setup_preview: "相机预览", setup_previewWait: "正在等待预览…", setup_previewError: "无法预览",
  setup_calHint: "在另一个终端运行缺失校准对应的命令，按 LeRobot 提示手动移动关节。",
  setup_calExtra: "若校准命令缺少依赖，请安装 lerobot[core-scripts]==0.7.0。",
  setup_checkAgain: "重新检查", setup_copy: "复制命令", setup_copied: "已复制命令。",
  setup_missing: "需要校准", setup_found: "已找到文件",
  setup_align: "启动采集前，请托住从臂、清空工作区域，并将两臂对齐。",
  setup_replace: "替换已有配置", setup_collect: "启动采集",
  setup_open: "打开采集控制台", setup_starting: "正在启动采集。请在服务就绪后打开控制台。",
  setup_loading: "正在扫描设备…", setup_working: "处理中…", setup_connect: "接入机械臂和相机后重新扫描。",
});

const steps = ["setup_arms", "setup_cameras", "setup_calibrate", "setup_review"];
let data, select = {}, stage = 0, files = [], result, busy = false, collection, previewTimer, message;
let dark = document.documentElement.dataset.theme === "dark" ||
  matchMedia("(prefers-color-scheme: dark)").matches;
try { dark = (localStorage.getItem("macbot.theme") || (dark ? "dark" : "light")) === "dark"; } catch (_) {}

function theme() {
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  $("theme").textContent = t(dark ? "setup_light" : "setup_dark");
}

function showMessage(text, translated = false, error = false) {
  message = text ? {text, translated, error} : null;
  const el = $("message");
  el.hidden = !message;
  el.textContent = message ? (translated ? t(text) : text) : "";
  el.classList.toggle("setup-error", error);
}

async function request(path, body) {
  const response = await fetch(path, body === undefined ? undefined : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Macbot-Setup": data.token},
    body: JSON.stringify({revision: data.revision, ...body}),
  });
  const value = await response.json();
  if (!response.ok || value.error) throw new Error(value.error || response.statusText);
  return value;
}

async function work(action, text = "setup_working") {
  if (busy) return;
  busy = true;
  syncButtons();
  showMessage(text, true);
  try {
    await action();
    if (message?.text === text) showMessage("");
  } catch (e) {
    showMessage(e.message, false, true);
  } finally {
    busy = false;
    syncButtons();
  }
}

function remember(inventory) {
  data = inventory;
  select = {family: "", cameras: {}, ...data.remembered};
  const external = data.cameras.map((cam, i) => [cam, i]).filter(([cam]) => !cam.builtin && cam.mode);
  if (!data.imported && !Object.keys(select.cameras).length && external.length === 1) select.cameras.top = external[0][1];
  stage = 0;
  result = null;
}

const option = (value, label, current) => `<option value="${esc(value)}"${value === current ? " selected" : ""}>${esc(label)}</option>`;
const portName = (d, i) => `${i + 1}. ${d.name}${d.serial ? ` · ${d.serial}` : d.port ? ` · ${d.port}` : ""}`;
const hex = x => x === null ? "—" : x.toString(16).padStart(4, "0");
const available = serial => data.devices.map((d, i) => [d, i]).filter(([d]) => Boolean(d.port) === serial);

function armSelect(role) {
  const serial = role === "leader" || select.family !== "rebot_b601";
  return `<label>${t("setup_" + role)}<select id="${role}">
    ${option("", t("setup_choose"), "")}
    ${available(serial).map(([d, i]) => option(i, portName(d, i), select[role])).join("")}</select></label>`;
}

function armStage() {
  const devices = data.devices.map((d, i) => `<li><strong>${esc(portName(d, i))}</strong>
    <span>${esc(d.suggestions.length ? d.adapter : t("setup_unknown"))} · ${hex(d.vid)}:${hex(d.pid)}
    · ${esc(d.serial || t("setup_noSerial"))}${d.suggestions.length ?
    " · " + esc(d.suggestions.map(k => data.families[k]).join(" / ")) : ""}</span></li>`).join("");
  const motion = select.family && select.family !== "rebot_b601" && available(true).length >= 2;
  return `<p>${t("setup_confirm")}</p><ul class="setup-inventory">${devices || `<li>${t("setup_noArms")}</li>`}</ul>
    <div class="setup-fields"><label>${t("setup_model")}<select id="family">
    ${option("", t("setup_choose"), select.family)}${Object.entries(data.families)
      .filter(([k]) => !data.imported || k === data.remembered.family).map(([k, v]) => option(k, v, select.family)).join("")}
    </select></label>${armSelect("leader")}${armSelect("follower")}</div>
    ${motion ? `<div class="btn-row"><button id="move" type="button">${t("setup_move")}</button>
      <span class="setup-note">${t("setup_moveHint")}</span></div>` : ""}`;
}

function cameraStage() {
  const keep = select.keep_framing !== false;
  const source = data.source_framing;
  const matches = data.camera_matches || {};
  const missing = Object.keys(source || {}).filter(role => !Object.hasOwn(matches, role));
  const choice = source ? `<p><label><input id="keep-framing" type="checkbox" ${keep ? "checked" : ""}> ${t("setup_keepFraming")}</label></p>
    ${!keep ? `<p class="setup-note">${t("setup_newFraming")}</p>` : missing.length ?
      `<p class="setup-note">${t("setup_unmatched")}: ${esc(missing.map(camLabel).join(", "))}</p>` : ""}` : "";
  const cards = data.cameras.map((c, i) => {
    const role = Object.keys(select.cameras).find(role => select.cameras[role] === i) || "";
    const original = keep && source?.[Object.keys(matches).find(r => matches[r] === i)];
    const detail = original ? `${t("setup_savedSize")} ${original.save_size.join(" × ")} · ` +
      (original.crop ? `${t("crop")} ${original.crop.join(", ")}` : t("setup_fullFrame")) :
      c.mode ? `${c.kind} · ${c.mode[0]} × ${c.mode[1]} · ${c.mode[2]} fps` : c.issue || "—";
    return `<article class="card"><h3>${esc(c.name)}</h3>
      <p class="setup-details">${esc(detail)}</p>
      <div class="view"><img alt="${esc(t("setup_preview"))}: ${esc(c.name)}" data-preview="${i}">
        <span class="tag">${t(c.mode ? "setup_previewWait" : "setup_previewError")}</span></div>
      <label class="setup-role">${t("setup_role")}<select data-camera="${i}" ${c.mode ? "" : "disabled"}>
        ${option("", t("setup_skip"), role)}${data.roles.map(r => option(r, camLabel(r), role)).join("")}
      </select></label></article>`;
  }).join("");
  return `<p>${t("setup_cameraHint")}</p>${choice}${cards ? `<div class="setup-cameras">${cards}</div>` : `<p>${t("setup_noCameras")}</p>`}
    ${data.notes.map(note => `<p class="setup-note">${esc(note)}</p>`).join("")}`;
}

function calibrationStage() {
  return `<p>${t("setup_calHint")}</p>${files.map((f, i) => `<div class="setup-cal">
    <h3>${t("setup_" + f.role)} <span class="badge ${f.ready ? "live" : "warn"}">${t(f.ready ? "setup_found" : "setup_missing")}</span></h3>
    <span class="setup-details mono">${esc(f.path)}</span>
    ${f.ready ? "" : `<pre>${esc(f.command)}</pre><button type="button" data-copy="${i}">${t("setup_copy")}</button>`}
    </div>`).join("")}${files.some(f => !f.ready) ? `<p class="setup-note">${t("setup_calExtra")}</p>` : ""}
    <button type="button" id="recheck">${t("setup_checkAgain")}</button>`;
}

function reviewStage() {
  return `<h3>${esc(data.families[select.family])}</h3>
    <p>${t("setup_leader")}: ${esc(portName(data.devices[select.leader], select.leader))}</p>
    <p>${t("setup_follower")}: ${esc(portName(data.devices[select.follower], select.follower))}</p>
    <p>${Object.entries(select.cameras).map(([role, i]) => `${esc(camLabel(role))}: ${esc(data.cameras[i].name)}`).join(" · ")}</p>
    ${data.exists ? `<p><label><input id="overwrite" type="checkbox" ${select.overwrite ? "checked" : ""}> ${t("setup_replace")}</label></p>` : ""}
    ${result ? `<p class="mono">${esc(result.path)}</p><pre>${esc(result.output)}</pre>` : ""}
    ${result?.ok ? `<p>${t("setup_align")}</p>` : ""}
    ${collection ? `<p><a href="http://127.0.0.1:${Number(collection.port)}/">${t("setup_open")}</a></p>` : ""}`;
}

function syncButtons() {
  $("rescan").disabled = !data || busy || !!collection;
  $("back").hidden = !data || stage === 0 || !!collection;
  $("back").disabled = busy;
  $("next").hidden = !data || !!collection;
  const allowed = [select.family && Number.isInteger(select.leader) && Number.isInteger(select.follower) &&
    select.leader !== select.follower, data.imported ? data.roles.every(r => Object.hasOwn(select.cameras, r)) :
    Object.keys(select.cameras || {}).length,
    files.length && files.every(f => f.ready), !data?.exists || select.overwrite || result?.ok];
  $("next").disabled = busy || !allowed[stage];
  $("next").textContent = t(stage !== 3 ? "setup_next" : result?.ok ? "setup_collect" : "setup_save");
  $("stage").querySelectorAll("select, input, button").forEach(el => {
    el.disabled = busy || el.dataset.camera !== undefined && !data.cameras[Number(el.dataset.camera)].mode;
  });
}

async function preview(img) {
  if (!img.isConnected || busy || stage !== 1) return;
  const url = `/api/preview/${img.dataset.preview}?revision=${data.revision}&keep=${select.keep_framing === false ? 0 : 1}`;
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error((await response.json()).error);
    const src = URL.createObjectURL(await response.blob());
    const old = img.src;
    img.src = src;
    if (old.startsWith("blob:")) URL.revokeObjectURL(old);
    img.nextElementSibling.textContent = t("live");
  } catch (e) {
    const old = img.src;
    img.removeAttribute("src");
    if (old.startsWith("blob:")) URL.revokeObjectURL(old);
    img.nextElementSibling.textContent = t("setup_previewError");
    img.nextElementSibling.title = e.message;
  }
}

function render(focus = false) {
  theme();
  if (!data) return;
  clearTimeout(previewTimer);
  $("stage").querySelectorAll("img[src^='blob:']").forEach(img => URL.revokeObjectURL(img.src));
  $("steps").setAttribute("aria-label", t("progress"));
  $("steps").innerHTML = steps.map((s, i) => `<li ${stage === i ? 'aria-current="step"' : i < stage ? 'class="done"' : ""}>
    <span>${i + 1}.</span>${t(s)}</li>`).join("");
  $("stage").innerHTML = `<h2 id="stage-title" tabindex="-1">${t(steps[stage])}</h2>` +
    [armStage, cameraStage, calibrationStage, reviewStage][stage]();
  if (focus) $("stage-title").focus();
  if (stage === 1) {
    const images = [...$("stage").querySelectorAll("[data-preview]")].filter(img => data.cameras[img.dataset.preview].mode);
    const tick = async () => {
      await Promise.all(images.map(preview));
      if (stage === 1 && images.some(img => img.isConnected)) previewTimer = setTimeout(tick, 400);
    };
    tick();
  }
  syncButtons();
}

async function checkFiles() {
  files = (await request("/api/calibration", select)).files;
}

$("stage").onchange = e => {
  const el = e.target;
  if (el.id === "family") {
    select.family = el.value;
    if (Boolean(data.devices[select.follower]?.port) === (el.value === "rebot_b601")) delete select.follower;
    if (el.value === "rebot_b601") {
      if (available(false).length === 1) select.follower = available(false)[0][1];
      if (available(true).length === 1) select.leader = available(true)[0][1];
    }
    render();
    $("family").focus();
  } else if (el.id === "leader" || el.id === "follower") {
    select[el.id] = el.value ? Number(el.value) : undefined;
  } else if (el.dataset.camera !== undefined) {
    const index = Number(el.dataset.camera);
    for (const role of Object.keys(select.cameras)) if (select.cameras[role] === index) delete select.cameras[role];
    if (el.value) select.cameras[el.value] = index;
    $("stage").querySelectorAll("[data-camera]").forEach(drop => {
      drop.value = Object.keys(select.cameras).find(r => select.cameras[r] === Number(drop.dataset.camera)) || "";
    });
  } else if (el.id === "keep-framing") {
    select.keep_framing = el.checked;
    render();
    $("keep-framing").focus();
  } else if (el.id === "overwrite") select.overwrite = el.checked;
  syncButtons();
};

$("stage").onclick = e => {
  const el = e.target.closest("button");
  if (!el) return;
  if (el.id === "move") {
    const serials = available(true).map(([, i]) => i);
    const candidates = serials.length === 2 ? serials : [select.leader, select.follower];
    if (!candidates.every(Number.isInteger) || candidates[0] === candidates[1]) return showMessage("setup_moveTwo", true, true);
    work(async () => {
      const reply = await request("/api/move", {...select, candidates});
      if (reply.leader === null) return showMessage("setup_moveFail", true);
      select.leader = reply.leader;
      select.follower = candidates.find(i => i !== reply.leader);
      render();
      showMessage("setup_moveOk", true);
    }, "setup_moving");
  } else if (el.id === "recheck") work(async () => { await checkFiles(); render(); $("recheck").focus(); });
  else if (el.dataset.copy !== undefined) work(async () => {
    await navigator.clipboard.writeText(files[Number(el.dataset.copy)].command);
    showMessage("setup_copied", true);
  });
};

$("back").onclick = () => { stage--; result = null; showMessage(""); render(true); };
$("next").onclick = () => work(async () => {
  if (stage === 0 || stage === 1) await checkFiles();
  if (stage < 3) stage++;
  else if (result?.ok) {
    collection = await request("/api/collect", {});
    showMessage("setup_starting", true);
  } else {
    result = await request("/api/save", select);
    if (!result.ok) { data.exists = true; select.overwrite = true; }
  }
  render(true);
});
$("rescan").onclick = () => work(async () => { remember(await request("/api/refresh", {})); render(true); }, "setup_loading");
$("theme").onclick = () => {
  dark = !dark;
  try { localStorage.setItem("macbot.theme", dark ? "dark" : "light"); } catch (_) {}
  theme();
};
window.onLang = () => {
  render();
  if (message) showMessage(message.text, message.translated, message.error);
};
theme();
work(async () => { remember(await request("/api/setup")); render(); }, "setup_loading");
