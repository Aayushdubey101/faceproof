// Page behaviour for all three pages. Every card is rendered by Jinja on the
// server, so this file only moves HTML around, animates progress and filters.

const POLL_MS = 1500;

function el(id) {
  return document.getElementById(id);
}

function toast(message, tone = "") {
  const node = document.createElement("div");
  node.className = `toast ${tone}`;
  node.textContent = message;
  el("toasts").append(node);
  setTimeout(() => node.remove(), 5000);
}

function show(id, html) {
  el(id).innerHTML = html || "";
  el(id).hidden = !html;
}

// the only card built client-side: a transport failure never reached the server
function errorCard(title, detail) {
  const escape = (value) =>
    String(value ?? "").replace(
      /[&<>"']/g,
      (character) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]
    );
  const lines = (detail || []).map((line) => `<p class="panel-note">${escape(line)}</p>`).join("");
  return `<section class="panel panel-bad">
      <h2 class="panel-title"><span class="panel-glyph">&#9888;</span>${escape(title || "Something went wrong")}</h2>
      ${lines || '<p class="panel-note">No further detail.</p>'}
    </section>`;
}

// copy buttons live inside server-rendered fragments, so listen at the document
document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy]");
  if (!button) return;
  await navigator.clipboard.writeText(button.dataset.copy);
  toast("Transaction hash copied.", "ok");
});

/* ---------------------------------------------------------------- run page */

if (el("run-form")) {
  const form = el("run-form");
  const runButton = el("run-button");
  const dropzone = el("dropzone");
  const probe = el("probe");
  let rendered = 0;
  let timer = null;

  function showProbe() {
    const [file] = probe.files;
    if (!file) return;
    el("probe-preview").src = URL.createObjectURL(file);
    el("probe-preview").hidden = false;
    el("dropzone-empty").hidden = true;
    el("file-name").textContent = file.name;
    el("file-size").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    el("upload-meta").hidden = false;
  }

  function clearProbe() {
    probe.value = "";
    el("probe-preview").hidden = true;
    el("probe-preview").removeAttribute("src");
    el("dropzone-empty").hidden = false;
    el("upload-meta").hidden = true;
  }

  probe.addEventListener("change", showProbe);
  el("replace-image").addEventListener("click", () => probe.click());
  el("remove-image").addEventListener("click", clearProbe);

  for (const name of ["dragenter", "dragover", "dragleave", "drop"]) {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.toggle("dragging", name === "dragenter" || name === "dragover");
    });
  }

  dropzone.addEventListener("drop", (event) => {
    if (!event.dataTransfer.files.length) return;
    probe.files = event.dataTransfer.files;
    showProbe();
  });

  function finish() {
    clearInterval(timer);
    runButton.disabled = false;
    runButton.textContent = "Run Investigation";
  }

  async function poll() {
    const status = await fetch(`/api/run?since=${rendered}`).then((response) => response.json());

    el("timeline").innerHTML = status.timeline_html;
    el("m-discovered").textContent = status.discovered;
    el("m-checked").textContent = status.checked;
    el("m-verified").textContent = status.verified;
    // measured on the server, not estimated from the candidate count
    el("runtime").textContent = `${status.elapsed.toFixed(1)}s`;

    // append only what is new, so the candidate thumbnails never reload mid-run
    if (status.count > rendered) {
      el("candidates").insertAdjacentHTML("beforeend", status.candidates_html);
      rendered = status.count;
    }
    show("best", status.best_html);

    if (status.state === "running") return;
    finish();
    show("error", status.error_html);
    show("result", status.result_html);

    if (status.error_html) toast("Investigation failed.", "bad");
    else if (status.result_html) toast("Evidence anchored on chain.", "ok");
    else toast("Investigation finished with no verified match.", "bad");
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!probe.files.length) return toast("Choose a probe image first.", "bad");

    rendered = 0;
    el("candidates").innerHTML = "";
    for (const id of ["error", "best", "result"]) show(id, "");
    for (const id of ["summary", "gallery-section"]) el(id).hidden = true;
    runButton.disabled = true;
    runButton.innerHTML = '<span class="spinner"></span>Running';

    const response = await fetch("/api/run", { method: "POST", body: new FormData(form) });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({ error: "Could not start the run." }));
      show("error", errorCard(payload.error, payload.detail));
      toast(payload.error || "Run failed.", "bad");
      finish();
      return;
    }
    el("runtime-line").hidden = false;
    el("progress").hidden = false;
    el("summary").hidden = false;
    el("gallery-section").hidden = false;
    timer = setInterval(poll, POLL_MS);
    poll();
  });
}

/* ------------------------------------------------------------- verify page */

if (el("file")) {
  // resolved per click, so the Restore Original button inside a rendered verdict works
  const allButtons = () => [...document.querySelectorAll("[data-tamper]")];

  const audit = async (button, tamper) => {
    const label = button.textContent;
    const buttons = allButtons();
    button.innerHTML = '<span class="spinner"></span>Checking';
    for (const one of buttons) one.disabled = true;
    show("error", "");
    show("result", "");

    const response = await fetch("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file: el("file").value, tamper }),
    });
    const payload = await response.json();

    if (!response.ok) {
      show("error", errorCard(payload.error, payload.detail));
      toast(payload.error || "Verification failed.", "bad");
    } else {
      show("result", payload.html);
      const ok = payload.ok && !payload.tampered;
      toast(
        ok ? "Fingerprint matches the anchor." : "Fingerprint does not match the anchor.",
        ok ? "ok" : "bad"
      );
    }

    button.textContent = label;
    for (const one of allButtons()) one.disabled = false;
  };

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-tamper]");
    if (button) audit(button, button.dataset.tamper === "1");
  });
}

/* ------------------------------------------------------------ archive page */

if (el("archive")) {
  const search = el("archive-search");
  const filters = el("archive-filters");
  const records = [...document.querySelectorAll(".record")];
  let filter = "all";

  const apply = () => {
    const query = search.value.trim().toLowerCase();
    let shown = 0;
    for (const record of records) {
      const data = record.dataset;
      const matchesFilter =
        filter === "all" ||
        (filter === "verified" && data.verified === "1" && data.tampered === "0") ||
        (filter === "tampered" && data.tampered === "1") ||
        (filter === "social" && data.social === "1");
      const visible = matchesFilter && data.text.includes(query);
      record.hidden = !visible;
      shown += visible ? 1 : 0;
    }
    el("archive-empty").hidden = shown > 0;
  };

  filters.addEventListener("click", (event) => {
    const button = event.target.closest(".filter");
    if (!button) return;
    for (const one of filters.children) one.classList.remove("active");
    button.classList.add("active");
    filter = button.dataset.filter;
    apply();
  });
  search.addEventListener("input", apply);
}
