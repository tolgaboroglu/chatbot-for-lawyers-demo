const EXAMPLES = {
  mevzuat: [
    "Kıdem tazminatına hangi durumlarda hak kazanılır?",
    "Üç yıldır çalışan işçinin ihbar süresi ne kadar?",
    "Konut kirasına yapılacak artış en fazla ne kadar olabilir?",
    "What is the statute of limitations for rent claims in Turkey?",
  ],
  ictihat: [
    "İstifa eden işçi kıdem tazminatı alabilir mi?",
    "Fazla mesai tanıkla ispatlanırsa ne olur?",
    "Is mediation mandatory before a reinstatement lawsuit?",
    "Boşanmada manevi tazminat için hangi şartlar aranıyor?",
  ],
};

const $ = (s) => document.querySelector(s);
const log = $("#log"), q = $("#q"), send = $("#send");
let mode = "mevzuat";
let history = [];
let counter = 0;

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function inline(s, id) {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/\[(\d{1,2})\]/g, (_, n) => `<a class="cite" href="#src-${id}-${n}">[${n}]</a>`);
}

function md(src, id) {
  const out = [];
  let list = null;
  const flush = () => { if (list) { out.push("<ul>" + list.join("") + "</ul>"); list = null; } };
  for (const raw of src.split("\n")) {
    const line = raw.trim();
    let m;
    if (!line) { flush(); continue; }
    if ((m = line.match(/^#{1,4}\s+(.*)/))) { flush(); out.push("<h3>" + inline(m[1].replace(/\*\*/g, ""), id) + "</h3>"); }
    else if ((m = line.match(/^(?:[-*•]|\d+\.)\s+(.*)/))) { (list ||= []).push("<li>" + inline(m[1], id) + "</li>"); }
    else { flush(); out.push("<p>" + inline(line, id) + "</p>"); }
  }
  flush();
  return out.join("");
}

function sourcesHtml(sources, id) {
  if (!sources.length) return "";
  const items = sources.map((s) => `
    <li class="src" id="src-${id}-${s.n}">
      <span class="n">[${s.n}]</span>
      <div><span class="t">${esc(s.title)}</span><span class="k ${s.kind}">${s.kind === "ictihat" ? "case law" : "legislation"}</span>
      <br><code>${esc(s.ref)}</code><p>${esc(s.snippet)}</p></div>
    </li>`).join("");
  return `<details class="sources" open><summary>${sources.length} sources used</summary><ol>${items}</ol></details>`;
}

function add(cls, html) {
  $("#empty")?.remove();
  const el = document.createElement("div");
  el.className = "msg " + cls;
  el.innerHTML = html;
  log.append(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

async function ask(question) {
  question = question.trim();
  if (!question || send.disabled) return;
  const id = ++counter;
  add("user", `<div class="bubble">${esc(question)}</div>`);
  const bot = add("bot", `<div class="who">Hukuk Asistanı</div><div class="answer"><p class="thinking">Searching sources…</p></div>`);
  send.disabled = true;
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode, history }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed.");
    bot.querySelector(".answer").innerHTML = md(data.answer, id);
    bot.insertAdjacentHTML("beforeend", sourcesHtml(data.sources, id));
    bot.querySelector(".who").textContent = `Hukuk Asistanı · ${data.models.generator}`;
    history.push({ role: "user", content: question }, { role: "assistant", content: data.answer });
    history = history.slice(-6);
  } catch (e) {
    bot.querySelector(".answer").innerHTML = `<p class="err">${esc(e.message)}</p>`;
  } finally {
    send.disabled = false;
    log.scrollTop = log.scrollHeight;
  }
}

function renderExamples() {
  const box = $("#examples");
  box.innerHTML = "";
  for (const text of EXAMPLES[mode]) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "ex";
    b.textContent = text;
    b.onclick = () => ask(text);
    box.append(b);
  }
}

document.querySelectorAll(".seg button").forEach((b) => {
  b.onclick = () => {
    mode = b.dataset.mode;
    document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    $("#chat-title").textContent = mode === "mevzuat" ? "Mevzuat Q&A" : "Case law";
    renderExamples();
  };
});

$("#clear").onclick = () => {
  history = [];
  log.innerHTML = `<div class="empty" id="empty"><h3>Ask about Turkish law</h3><p>Pick an example on the left or type your own question.</p></div>`;
};

$("#form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = q.value;
  q.value = "";
  q.style.height = "auto";
  ask(text);
});
q.addEventListener("input", () => { q.style.height = "auto"; q.style.height = Math.min(q.scrollHeight, 160) + "px"; });
q.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#form").requestSubmit(); }
});

renderExamples();
