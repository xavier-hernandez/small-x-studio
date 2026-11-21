// "?" help bubbles: hover or keyboard focus shows them; a tap or click toggles them (for touch screens).
// preventDefault stops the click from also focusing the field the label belongs to.
export function initHelp() {
  document.querySelectorAll(".help").forEach(help => {
    help.addEventListener("click", e => {
      e.preventDefault();
      const open = !help.classList.contains("open");
      document.querySelectorAll(".help.open").forEach(h => h.classList.remove("open"));
      help.classList.toggle("open", open);
    });
    help.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); help.click(); }
      else if (e.key === "Escape") { help.classList.remove("open"); help.blur(); }
    });
  });
  document.addEventListener("click", e => {
    if (!e.target.closest(".help")) document.querySelectorAll(".help.open").forEach(h => h.classList.remove("open"));
  });
}

/**
 * Polls /api/status until every model is ready or failed, and shows one model's status in `el`.
 * @param getId the model's id, or a function returning the currently selected one
 * @param onChange called with that model's { phase, detail }, or null if it isn't known (site unreachable)
 * @returns {{ show() }} call show() after the selection changes, to update `el` straight away
 */
export function watchModel(getId, el, onChange) {
  let models = null, reachable = true;
  const show = () => {
    const s = models?.[typeof getId === "function" ? getId() : getId] ?? null;
    el.className = "status" + (s?.phase === "ready" ? " ready" : s?.phase === "error" ? " error" : "");
    el.textContent = !reachable ? "Site unreachable"
      : !models ? "Connecting…"
      : !s ? "This model isn't set up on the server"
      : s.phase === "ready" ? "Model available"
      : s.phase === "error" ? `Model unavailable · ${s.detail}`
      : s.phase === "waiting" ? "Site will fetch this model after the current downloads"
      : `Site is fetching the model${s.detail ? " · " + s.detail : ""}`;
    onChange?.(s);
  };
  const poll = async () => {
    try { models = (await (await fetch("/api/status")).json()).models ?? {}; reachable = true; } catch { reachable = false; }
    show();
    const pending = !reachable || Object.values(models ?? {}).some(s => s.phase !== "ready" && s.phase !== "error");
    if (pending) setTimeout(poll, 1500);
  };
  poll();
  return { show };
}
