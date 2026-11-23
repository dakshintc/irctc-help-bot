(() => {
  "use strict";

  const chat = document.getElementById("chat");
  const form = document.getElementById("composer");
  const input = document.getElementById("message");
  const sendBtn = document.getElementById("send");
  const langSel = document.getElementById("lang");
  const clearBtn = document.getElementById("clear");
  const suggestions = document.getElementById("suggestions");

  let history = [];
  let busy = false;

  // Restore the language choice for this browser.
  try {
    const saved = localStorage.getItem("irctc-help-lang");
    if (saved && ["auto", "en", "hi"].includes(saved)) langSel.value = saved;
  } catch (_) {}

  langSel.addEventListener("change", () => {
    try { localStorage.setItem("irctc-help-lang", langSel.value); } catch (_) {}
  });

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function formatBubble(bubble, text) {
    const lines = String(text || "").split(/\n+/);
    let list = null;
    let paragraph = null;

    const flushParagraph = () => {
      if (paragraph && paragraph.textContent.trim()) bubble.appendChild(paragraph);
      paragraph = null;
    };

    lines.forEach((raw) => {
      const line = raw.trim();
      if (!line) return;
      if (/^[-•*]\s+/.test(line)) {
        flushParagraph();
        if (!list) {
          list = el("ul");
          bubble.appendChild(list);
        }
        list.appendChild(el("li", null, line.replace(/^[-•*]\s+/, "")));
      } else {
        if (list) list = null;
        if (!paragraph) paragraph = el("p");
        paragraph.textContent += (paragraph.textContent ? " " : "") + line;
      }
    });
    flushParagraph();
    if (!bubble.childNodes.length) bubble.textContent = text;
  }

  function addMessage(role, text, options = {}) {
    const wrap = el("div", "message " + role);
    const avatar = el("div", "avatar", role === "user" ? "You" : "IR");
    avatar.setAttribute("aria-hidden", "true");
    const body = el("div", "body");
    const bubble = el("div", "bubble" + (options.warn ? " warn" : ""));

    if (options.typing) {
      const dots = el("div", "typing");
      dots.setAttribute("aria-label", "Assistant is typing");
      dots.innerHTML = "<i></i><i></i><i></i>";
      bubble.appendChild(dots);
    } else {
      formatBubble(bubble, text);
    }

    body.appendChild(bubble);

    if (options.sources && options.sources.length) {
      const row = el("div", "sources");
      options.sources.forEach((source) => {
        const link = el("a", "source");
        link.href = source.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.title = source.url;
        link.appendChild(el("span", null, source.title || "IRCTC page"));
        link.appendChild(document.createTextNode(" ↗"));
        row.appendChild(link);
      });
      body.appendChild(row);
    }

    if (options.meta) {
      body.appendChild(el("div", "meta", options.meta));
    }

    wrap.appendChild(avatar);
    wrap.appendChild(body);
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
    return wrap;
  }

  function setBusy(value) {
    busy = value;
    sendBtn.disabled = value;
    input.disabled = value;
    chat.setAttribute("aria-busy", value ? "true" : "false");
  }

  async function ask(question) {
    const text = String(question || "").trim();
    if (!text || busy) return;

    suggestions.style.display = "none";
    addMessage("user", text);
    history.push({ role: "user", content: text });

    input.value = "";
    input.style.height = "auto";
    setBusy(true);
    const typing = addMessage("bot", "", { typing: true });

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 90000);

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          lang: langSel.value,
          history: history.slice(-6, -1),
        }),
        signal: controller.signal,
      });

      let data = {};
      try { data = await response.json(); } catch (_) {}

      typing.remove();

      if (!response.ok) {
        const message = data.message || "Could not get a reply. Please try again.";
        addMessage("bot", message, { warn: true });

        if (data.error === "sensitive_data" && !history.length) history = [];
        else if (data.error === "sensitive_data") {
          // Drop the blocked message from history; it must not be replayed.
          history = history.filter((turn) => turn.content !== text);
        }
        return;
      }

      const answer = data.answer || "I'm not sure — please check irctc.co.in.";
      const metaParts = [];
      if (data.mode === "offline" || data.mode === "offline-fallback") {
        metaParts.push("from the local FAQ index");
      } else if (data.mode === "claude") {
        metaParts.push("checked against IRCTC pages");
      }
      if (data.confident === false) metaParts.push("low confidence");

      addMessage("bot", answer, {
        sources: data.sources || [],
        meta: metaParts.join(" · "),
        warn: Boolean(data.blocked),
      });
      history.push({ role: "assistant", content: answer });
    } catch (error) {
      typing.remove();
      const msg = error.name === "AbortError"
        ? "The reply took too long. Please try again."
        : "Network error. Check that the Flask server is running and try again.";
      addMessage("bot", msg, { warn: true });
    } finally {
      clearTimeout(timer);
      setBusy(false);
      input.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    ask(input.value);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ask(input.value);
    }
  });

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 130) + "px";
  });

  suggestions.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-q]");
    if (!button) return;
    ask(button.getAttribute("data-q"));
  });

  clearBtn.addEventListener("click", () => {
    chat.querySelectorAll(".message").forEach((node, index) => {
      if (index > 0) node.remove();
    });
    history = [];
    suggestions.style.display = "";
    const sources = chat.querySelectorAll(".sources");
    sources.forEach((node) => node.remove());
    input.focus();
  });

  input.focus();
})();
