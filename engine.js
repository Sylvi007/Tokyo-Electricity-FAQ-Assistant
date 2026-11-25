// JavaScript port of qa_engine.py (same data, rules and TF-IDF maths as scikit-learn's TfidfVectorizer).
function makeEngine(D) {
  const SYN = D.synonyms.map(([p, r]) => [new RegExp(`\\b(?:${p})(?=\\b|\\s|$)`, "g"), r]);
  const STOP = new Set(D.stop), DOMAIN = new Set(D.domain);

  function normalise(text) {
    let t = text.toLowerCase().replace(/[“”"'’`]/g, " ");
    for (const [re, r] of SYN) t = t.replace(re, r);
    t = t.replace(/[^\p{L}\p{N}_\s%°]/gu, " ");
    return t.split(/\s+/).filter(w => w && !STOP.has(w)).join(" ");
  }
  const wordTokens = t => t.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || [];
  function wordNgrams(t, maxN) {
    const tok = wordTokens(t), out = [...tok];
    if (maxN >= 2) for (let i = 0; i + 1 < tok.length; i++) out.push(tok[i] + " " + tok[i + 1]);
    return out;
  }
  function charWb(t) {
    const out = [];
    for (let w of t.toLowerCase().split(/\s+/).filter(Boolean)) {
      w = " " + w + " ";
      for (let n = 3; n <= 5; n++) {
        let off = 0;
        out.push(w.slice(off, off + n));
        while (off + n < w.length) { off++; out.push(w.slice(off, off + n)); }
        if (off === 0) break;
      }
    }
    return out;
  }
  function tfidf(docs, analyzer) {
    const vocab = new Map(), df = new Map();
    const counted = docs.map(d => { const c = new Map(); for (const g of analyzer(d)) c.set(g, (c.get(g) || 0) + 1); return c; });
    for (const c of counted) for (const g of c.keys()) df.set(g, (df.get(g) || 0) + 1);
    const n = docs.length, idf = new Map();
    for (const [g, f] of df) idf.set(g, Math.log((1 + n) / (1 + f)) + 1);
    const vec = c => {
      const v = new Map(); let norm = 0;
      for (const [g, k] of c) if (idf.has(g)) { const x = (1 + Math.log(k)) * idf.get(g); v.set(g, x); norm += x * x; }
      norm = Math.sqrt(norm) || 1; for (const [g, x] of v) v.set(g, x / norm); return v;
    };
    const rows = counted.map(vec);
    return { rows, transform: d => { const c = new Map(); for (const g of analyzer(d)) c.set(g, (c.get(g) || 0) + 1); return vec(c); } };
  }
  const dot = (a, b) => { let s = 0; for (const [g, x] of a) { const y = b.get(g); if (y) s += x * y; } return s; };

  const texts = [], owner = [];
  D.faq.forEach((f, i) => [f.question, ...(f.variants || [])].forEach(q => { texts.push(normalise(q)); owner.push(i); }));
  const W = tfidf(texts, t => wordNgrams(t, 2)), C = tfidf(texts, charWb);
  const A = tfidf(D.faq.map(f => normalise(f.answer.replace(/\*/g, ""))), t => wordNgrams(t, 1));

  function scores(text) {
    const q = normalise(text), qw = q.replace(/\b\d+\b/g, " ");
    const wv = W.transform(qw), cv = C.transform(q), av = A.transform(qw);
    const pw = new Array(D.faq.length).fill(0), pc = new Array(D.faq.length).fill(0);
    W.rows.forEach((r, k) => { pw[owner[k]] = Math.max(pw[owner[k]], dot(wv, r)); });
    C.rows.forEach((r, k) => { pc[owner[k]] = Math.max(pc[owner[k]], dot(cv, r)); });
    const total = D.faq.map((_, i) => D.weights.word * pw[i] + D.weights.char * pc[i] + D.weights.answer * dot(av, A.rows[i]));
    return { total, wordBest: Math.max(...pw) };
  }

  function validate(text) {
    if (text == null || !String(text).trim()) return "Please type a question first. For example: *When is electricity demand highest?*";
    const t = String(text).trim();
    if (t.length > D.maxChars) return `That's ${t.length} characters. Please keep your question under ${D.maxChars} characters.`;
    const letters = (t.match(/[A-Za-z]/g) || []).length;
    if (/[぀-ヿ一-鿿]/.test(t) && letters < 3) return "Sorry, I can only understand questions in English for now (日本語は未対応です).";
    if (letters < 3) return "That doesn't look like a question. Please use words, e.g. *What was the record demand?*";
    if (new Set(t.toLowerCase().replace(/ /g, "")).size <= 2) return "That doesn't look like a question. Please try again with a full sentence.";
    return null;
  }

  function findDate(text) {
    const t = text.toLowerCase(); let y, mo, d, m;
    if ((m = t.match(/\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b/))) [y, mo, d] = m.slice(1).map(Number);
    else {
      const mon = Object.keys(D.months).sort((a, b) => b.length - a.length).join("|");
      const m1 = t.match(new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(${mon})\\.?,?\\s+(20\\d{2})\\b`));
      const m2 = t.match(new RegExp(`\\b(${mon})\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?,?\\s+(20\\d{2})\\b`));
      if (m1) [d, mo, y] = [+m1[1], D.months[m1[2]], +m1[3]];
      else if (m2) [mo, d, y] = [D.months[m2[1]], +m2[2], +m2[3]];
      else return null;
    }
    const dt = new Date(Date.UTC(y, mo - 1, d));
    if (dt.getUTCFullYear() !== y || dt.getUTCMonth() !== mo - 1 || dt.getUTCDate() !== d) return "invalid";
    return dt;
  }

  const days = Object.keys(D.daily).sort();
  const fmt = (dt, opts) => dt.toLocaleDateString("en-GB", { timeZone: "UTC", ...opts });
  function answerDate(dt) {
    if (dt === "invalid") return { status: "invalid", message: "That date doesn't exist. Please check the day and month." };
    const key = dt.toISOString().slice(0, 10), first = new Date(days[0]), last = new Date(days[days.length - 1]);
    const short = x => fmt(x, { day: "2-digit", month: "short", year: "numeric" });
    if (!D.daily[key]) return { status: "unsure", message: `I only have data from **${short(first)}** to **${short(last)}**. ${short(dt)} is outside that range.`, related: [] };
    const [peak, ph, low, mean, tmax, tmin, tmean, work, season] = D.daily[key];
    const diff = peak - D.avgPeak;
    const message = `**${fmt(dt, { weekday: "long", day: "2-digit", month: "long", year: "numeric" })}** (${work ? "a workday" : "a weekend/holiday"}, ${season.toLowerCase()}):\n\n` +
      `- Peak demand: **${peak.toFixed(1)} GW** at ${String(ph).padStart(2, "0")}:00 (${Math.abs(diff).toFixed(1)} GW ${diff >= 0 ? "above" : "below"} the average daily peak of ${D.avgPeak.toFixed(1)} GW)\n` +
      `- Lowest demand: **${low.toFixed(1)} GW**, average ${mean.toFixed(1)} GW\n` +
      `- Tokyo temperature: **${tmin.toFixed(1)} – ${tmax.toFixed(1)} °C** (mean ${tmean.toFixed(1)} °C)`;
    return { status: "date", message, confidence: 1, matched: "Date lookup in the real data", category: "Data lookup", related: [] };
  }

  function ask(text, threshold = D.threshold) {
    const err = validate(text);
    if (err) return { status: "invalid", message: err, related: [] };
    text = String(text).trim();
    const dt = findDate(text);
    if (dt !== null) return answerDate(dt);
    const { total: s, wordBest } = scores(text);
    const order = s.map((v, i) => i).sort((a, b) => s[b] - s[a]);
    const best = order[0];
    const related = order.slice(1, 4).filter(i => s[i] >= threshold * 0.6).map(i => D.faq[i].question);
    const inDomain = normalise(text).split(" ").some(w => DOMAIN.has(w));
    if (s[best] < threshold || wordBest < D.gate || !inDomain)
      return { status: "unsure", confidence: s[best], related: order.slice(0, 3).map(i => D.faq[i].question),
               message: "Sorry, I'm not sure I understood. I can answer questions about Tokyo electricity demand, weather, solar power and this project's data. Did you mean one of these?" };
    const f = D.faq[best];
    return { status: "answer", message: f.answer, confidence: s[best], matched: f.question, category: f.category, id: f.id, related };
  }
  return { ask, scores, normalise, faq: D.faq };
}
if (typeof module !== "undefined") module.exports = { makeEngine };
