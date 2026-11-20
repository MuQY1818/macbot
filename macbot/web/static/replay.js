// The run and standalone replay consoles share this viewer and the rollout routes.
async function replayFetch(url, options) {
  const r = await fetch(url, options);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || "replay_failed");
  return r;
}

const rolloutURL = n => `/api/rollout/${encodeURIComponent(n)}`;
const isEditing = e => e.target.closest("input,select,textarea,button,[contenteditable]");

class RolloutViewer {
  constructor({ status, change, result, trialHeading = false }) {
    this.status = status;
    this.change = change;
    this.result = result;
    this.trialHeading = trialHeading;
    if (trialHeading) delete $("rinfo-title").dataset.i18n;
    $("rplay").onclick = () => this.togglePlay();
    $("rspeed").onchange = () => { if (this.timer) { this.stop(); this.togglePlay(); } };
    $("rslider").oninput = e => { this.stop(); this.seek(+e.target.value); };
    $("rplot").onclick = e => {
      if (!this.data) return;
      const b = e.currentTarget.getBoundingClientRect();
      this.stop();
      this.seek(this.frameAt(((e.clientX - b.left) / b.width) * this.duration));
    };
    $("rreview").onsubmit = e => { e.preventDefault(); this.save(); };
  }

  close() {
    this.stop();
    this.load?.abort();
    this.load = this.data = this.name = null;
    this.wanted = this.shown = null;
    $("rviews").replaceChildren();
    $("rframe-error").hidden = true;
    $("rreview-msg").textContent = "";
    this.reviewMessage = null;
    this.change(false);
  }

  async open(name) {
    this.close();
    const load = this.load = new AbortController();
    this.name = name;
    this.status("replay_loading");
    try {
      const r = await (await replayFetch(rolloutURL(name), { signal: load.signal })).json();
      if (load.signal.aborted) return;
      if (!r.t.length) throw new Error("replay_missing");
      this.data = r;
      this.duration = r.t.at(-1);
      layoutViews($("rviews"), r.cams, c => r.sizes[c], "r-");
      $("rslider").max = r.t.length - 1;
      $("rslider").value = 0;
      $("rpos").textContent = "";
      $("rchunk").textContent = "";
      $("rresult").value = r.meta.result || "failure";
      $("rnote").value = r.meta.note || "";
      $("rsave").disabled = false;
      this.translate();
      this.status(null);
      this.change(true);
      this.seek(0);
    } catch (e) {
      if (!load.signal.aborted) this.status(e.message in I18N.en ? e.message : "replay_failed");
    }
  }

  translate() {
    $("rplay").firstElementChild.textContent = t(this.timer ? "pause" : "play");
    if (this.reviewMessage) $("rreview-msg").textContent = t(this.reviewMessage);
    if (!$("rframe-error").hidden) $("rframe-error").textContent = t(this.frameError);
    if (!this.data) return;
    const r = this.data;
    const fraction = ts => this.duration > 0 ? Math.max(0, Math.min(1, ts / this.duration)) : 0;
    const marks = r.chunk_t.map((ts, i) => {
      const x = (1000 * fraction(ts)).toFixed(1);
      return `<line x1="${x}" x2="${x}" y1="182" y2="190" stroke="currentColor" opacity=".45" vector-effect="non-scaling-stroke">` +
        `<title>${t("chunk_at", i + 1)} · ${ts.toFixed(2)} s · ${r.chunk_ms[i] ?? "–"} ms</title></line>`;
    }).join("");
    drawTraces($("rplot"), $("rlegend"), r.q, r.cmd, r.meta.joint_names, marks, r.t.map(fraction));
    $("rlegend").innerHTML += `<span style="--c:transparent">${t("chunk_marks")}</span>`;
    for (const c of r.cams) $("r-tag-" + c).textContent = `${camLabel(c)} · ${t("replay")}`;
    if (this.shown != null) this.position(this.shown);
    this.info();
  }

  info() {
    const r = this.data, m = r.meta;
    $("rtitle").textContent = `${this.name} · ${this.duration.toFixed(1)} s`;
    if (this.trialHeading) $("rinfo-title").textContent = t("trial_heading", this.name);
    const tile = (key, value, unit = "") => `<div class="tile"><div class="k">${t(key)}</div><div class="v">${esc(value)}<small> ${unit}</small></div></div>`;
    $("rtiles").innerHTML = tile("length", this.duration.toFixed(1), "s") + tile("chunks", r.chunks) +
      tile("median_infer", r.infer_ms == null ? "–" : Math.round(r.infer_ms), "ms") + tile("speed", `×${m.speed ?? "–"}`);
    const values = { policy: String(m.policy || "").split("/").slice(-2).join("/"),
      executor: m.executor || m.mode || "", th_saved: m.saved_at || "", instruction: m.instruction || "" };
    $("rinfo").innerHTML = Object.entries(values).map(([key, value]) => `<dt>${t(key)}</dt><dd>${esc(value)}</dd>`).join("");
  }

  frameAt(time) {
    const ts = this.data.t;
    let lo = 0, hi = ts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (ts[mid] < time) lo = mid + 1; else hi = mid;
    }
    return lo > 0 && time - ts[lo - 1] < ts[lo] - time ? lo - 1 : lo;
  }

  position(k) {
    const r = this.data, ts = r.t[k];
    $("rpos").textContent = `${ts.toFixed(2)} s`;
    moveCursor($("rplot"), k, r.t.length, this.duration ? ts / this.duration : 0);
    let i = r.chunk_t.findIndex(x => x > ts);
    i = (i === -1 ? r.chunk_t.length : i) - 1;
    $("rchunk").textContent = i >= 0 ? `${t("chunk_at", i + 1)} · ${r.chunk_ms[i] ?? "–"} ms` : "";
  }

  seek(k) {
    if (!this.data) return;
    this.wanted = Math.max(0, Math.min(k, this.data.t.length - 1));
    $("rslider").value = this.wanted;
    if (this.pending !== this.load && (this.shown !== this.wanted || !$("rframe-error").hidden)) this.showFrames();
  }

  async showFrames() {
    const load = this.pending = this.load, {signal} = load;
    const k = this.wanted, base = rolloutURL(this.name), urls = [];
    try {
      const frames = await Promise.allSettled(this.data.cams.map(async c => {
        const r = await replayFetch(`${base}/${encodeURIComponent(c)}/${k}`, { signal });
        const img = new Image(), url = URL.createObjectURL(await r.blob());
        urls.push(url);
        img.src = url;
        await img.decode();
        return [c, img];
      }));
      if (signal.aborted) return;
      const failed = frames.find(f => f.status === "rejected");
      if (failed) throw failed.reason;
      // Commit one decoded set together; a slow camera never leaves the views on different frames.
      for (const { value: [c, img] } of frames) {
        const old = $("r-cam-" + c);
        img.id = old.id;
        img.alt = old.alt;
        old.replaceWith(img);
      }
      $("rframe-error").hidden = true;
      this.shown = k;
      this.position(k);
    } catch (e) {
      if (!signal.aborted) {
        this.stop();
        this.frameError = e.message in I18N.en ? e.message : "replay_failed";
        $("rframe-error").textContent = t(this.frameError);
        $("rframe-error").hidden = false;
      }
    } finally {
      for (const url of urls) URL.revokeObjectURL(url);
      if (this.pending === load) this.pending = null;
      if (!signal.aborted && this.wanted !== k) this.showFrames();
    }
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
    $("rplay").firstElementChild.textContent = t("play");
  }

  togglePlay() {
    if (!this.data) return;
    if (this.timer) return this.stop();
    const start = this.wanted == null || this.wanted >= this.data.t.length - 1 ? 0 : this.wanted;
    this.seek(start);
    const ts = this.data.t[start], wall = performance.now(), speed = +$("rspeed").value;
    $("rplay").firstElementChild.textContent = t("pause");
    this.timer = setInterval(() => {
      const time = ts + (performance.now() - wall) * speed / 1000;
      this.seek(this.frameAt(time));
      if (time >= this.duration) this.stop();
    }, 1000 / 15);
  }

  key(e) {
    if (!this.data || isEditing(e)) return false;
    if (e.key === " ") this.togglePlay();
    else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      this.stop();
      this.seek((this.wanted ?? 0) + (e.key === "ArrowRight" ? 1 : -1));
    } else return false;
    e.preventDefault();
    return true;
  }

  async save() {
    if (!this.data || $("rsave").disabled) return;
    const { signal } = this.load, name = this.name, r = this.data;
    $("rsave").disabled = true;
    try {
      const review = await (await replayFetch(rolloutURL(name) + "/annotation", {
        method: "POST", headers: { "Content-Type": "application/json" }, signal,
        body: JSON.stringify({ result: $("rresult").value, note: $("rnote").value }),
      })).json();
      if (signal.aborted) return;
      Object.assign(r.meta, review);
      this.result(name, review.result);
      this.reviewMessage = "review_saved";
    } catch (e) {
      if (!signal.aborted) this.reviewMessage = e.message in I18N.en ? e.message : "review_failed";
    } finally {
      if (!signal.aborted) {
        $("rsave").disabled = false;
        $("rreview-msg").textContent = t(this.reviewMessage);
      }
    }
  }
}

class RolloutList {
  constructor(body, open, { toggle = null, timed = false } = {}) {
    this.body = body;
    this.open = open;
    this.toggle = toggle;
    this.timed = timed;
    this.items = [];
    this.enabled = true;
    if (toggle) toggle.onclick = () => { this.expanded = !this.expanded; this.render(); };
    body.onclick = e => {
      const row = e.target.closest("tr[data-n]");
      if (row && this.enabled) this.open(row.dataset.n);
    };
  }

  async read(signal) {
    return (await (await replayFetch("/api/rollouts", { signal })).json()).items;
  }

  updateResult(name, result) {
    const item = this.items.find(x => x.name === name);
    if (item) item.result = result;
    this.render();
  }

  render() {
    const items = !this.toggle || this.expanded ? this.items : this.items.slice(0, 5);
    // Keep focus and pointer targets stable during the run console's status poll.
    const key = JSON.stringify([LANG, this.enabled, this.selected, items, this.items.length, this.expanded]);
    if (key === this.key) return;
    this.key = key;
    this.body.closest("table").setAttribute("aria-disabled", !this.enabled);
    this.body.innerHTML = items.length ? items.map(r => {
      const id = esc(r.name.split("_")[1]), ok = r.result === "success";
      const result = `<td><span class="pill ${ok ? "ok" : "bad"}">${t(ok ? "success" : "failure")}</span></td>`;
      const disabled = this.enabled ? "" : " disabled";
      const first = this.timed ? `<button class="ghost num" aria-label="${t("replay_btn")} ${id}"${disabled}>${id}</button>` : id;
      const last = this.timed ? `<td class="num saved-time">${esc(r.saved_at || "–")}</td>` :
        `<td class="acts"><button${disabled}>${t("replay_btn")}</button></td>`;
      return `<tr data-n="${esc(r.name)}" class="${r.name === this.selected ? "sel" : ""}"><td class="num">${first}</td>` +
        `${result}<td class="num rollout-duration">${esc(r.s)}&nbsp;s</td>${last}</tr>`;
    }).join("") : `<tr><td colspan="4" class="empty">${t(this.timed ? "no_trials" : "no_rollouts")}</td></tr>`;
    if (this.toggle) {
      this.toggle.hidden = this.items.length <= 5;
      this.toggle.textContent = this.expanded ? t("show_less") : t("show_all", this.items.length);
      this.toggle.setAttribute("aria-expanded", !!this.expanded);
    }
  }
}
