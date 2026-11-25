"""
NLP question-answering engine for the Tokyo Electricity FAQ Assistant.

Pipeline for every question:
  1. Validate      - reject empty, too short, too long or meaningless input with a helpful message
  2. Date lookup   - if the question contains a date, answer it from the real daily data
  3. Normalise     - lower-case, unify units and synonyms ("power usage" -> "demand", "hottest" -> "hot")
  4. Vectorise     - TF-IDF on words (1-2 grams) + TF-IDF on character 3-5 grams (robust to typos)
  5. Match         - cosine similarity against every known question and its paraphrases
  6. Decide        - best answer if similarity >= threshold, otherwise "not sure" + suggestions

Only scikit-learn, pandas and the Python standard library are used. No internet or external AI service is needed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
FAQ_PATH = HERE / "faq.json"
DAILY_PATH = HERE / "tokyo_daily_summary.csv"

DEFAULT_THRESHOLD = 0.30
DOMAIN_GATE = 0.12      # minimum word-level similarity: the question must share real words with the knowledge base
# Out-of-scope detector: a question must mention at least one term from this domain (after normalisation).
DOMAIN_TERMS = {
    "demand", "electricity", "electric", "power", "energy", "grid", "gw", "mw", "kw", "gwh", "kwh", "万kw", "watt",
    "watts", "tepco", "jma", "weather", "temperature", "hot", "cold", "humidity", "solar", "supply", "capacity",
    "blackout", "shortage", "outage", "peak", "lowest", "average", "workday", "weekend", "holiday", "holidays",
    "obon", "golden", "silver", "bridge", "season", "summer", "winter", "spring", "autumn", "month", "hour",
    "data", "dataset", "date", "day", "forecast", "model", "feature", "features", "station", "kumagaya",
    "climate", "air", "conditioning", "heating", "cooling", "save", "saving", "節電", "usage", "rate", "net",
    "duck", "curve", "trend", "year", "cleaning", "missing", "chatbot", "assistant", "nlp", "kanto", "tokyo",
    "plants", "plant", "appliances", "washing", "dryer", "japan", "meteorological"}
WEIGHTS = {"word": 0.55, "char": 0.30, "answer": 0.15}
MAX_CHARS = 300

# Words that mean the same thing in this domain -> one canonical word (applied as whole-word replacements).
SYNONYMS = {
    r"solar power|solar energy|solar electricity": "solar",
    r"electricity (usage|use|consumption)": "demand", r"power (usage|use|consumption)": "demand",
    r"energy (usage|use|consumption)": "demand", r"electricity demand": "demand", r"power demand": "demand",
    r"load": "demand", r"consumption": "demand", r"usage": "demand",
    r"muggy|humid|humidity|sticky": "humidity", r"forecasting|forecasts|forecasted|predicting|predicts|"
    r"prediction|predictions|predict|precise|accurate|accuracy": "forecast",
    r"busiest|busy": "peak", r"quietest|quiet|calmest": "lowest",
    r"years|yearly|annual|annually|year": "year",
    r"hottest|hotter|heat ?wave|heat": "hot", r"coldest|colder|chilly|freezing": "cold",
    r"temp|temps|temperatures|degrees|°c|celsius": "temperature",
    r"highest|maximum|max|biggest|largest|most|record|peak": "peak",
    r"lowest|minimum|min|smallest|least|fewest": "lowest",
    r"weekend|weekends|saturday|saturdays|sunday|sundays": "weekend",
    r"weekday|weekdays|workday|workdays|business day": "workday",
    r"photovoltaic|pv|solar panels?|sun power": "solar",
    r"gigawatts?": "gw", r"megawatts?": "mw", r"kilowatts?": "kw",
    r"a\.?i\.?|machine learning|ml model|algorithm": "model",
    r"tokyo electric power( company)?": "tepco", r"japan meteorological agency|meteorological agency": "jma",
}
STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "of", "in", "on", "at", "to", "for", "and", "or",
              "it", "this", "that", "do", "does", "did", "i", "you", "me", "my", "your", "can", "please", "tell",
              "about", "there", "with", "by", "s", "what", "which", "how", "much", "many", "who", "whom", "whose",
              "why", "where", "when", "good", "best", "better", "get", "got", "have", "has", "had", "will", "would",
              "should", "could", "any", "some", "all", "very", "really", "more", "than", "give", "buy", "make",
              "know", "want", "like", "just", "its", "their", "they", "we", "our", "us", "from", "if", "so",
              "into", "out", "up", "down", "over", "same", "also", "here", "ever", "one", "new", "now"}

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in MONTHS.items()})
MONTHS["sept"] = 9


def normalise(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[“”\"'’`]", " ", t)
    for pattern, repl in SYNONYMS.items():
        t = re.sub(rf"\b(?:{pattern})(?=\b|\s|$)", repl, t)
    t = re.sub(r"[^\w\s%°]", " ", t)
    return " ".join(w for w in t.split() if w not in STOP_WORDS)


@dataclass
class Result:
    status: str                    # "answer", "date", "unsure", "invalid"
    message: str                   # text to show the user
    confidence: float = 0.0        # 0-1
    matched_question: str = ""
    category: str = ""
    related: list = field(default_factory=list)


class FAQAssistant:
    def __init__(self, faq_path: Path = FAQ_PATH, daily_path: Path = DAILY_PATH,
                 threshold: float = DEFAULT_THRESHOLD):
        self.faq = json.loads(Path(faq_path).read_text(encoding="utf-8"))
        self.threshold = threshold
        self.daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date") if Path(daily_path).exists() \
            else None
        # One row per phrasing; remember which FAQ entry it belongs to.
        self.texts, self.owner = [], []
        for i, item in enumerate(self.faq):
            for q in [item["question"], *item.get("variants", [])]:
                self.texts.append(normalise(q))
                self.owner.append(i)
        self.owner = np.array(self.owner)
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.word_matrix = self.word_vec.fit_transform(self.texts)
        self.char_matrix = self.char_vec.fit_transform(self.texts)
        # The answers are a second, weaker source of words (e.g. a question about "air-conditioning").
        self.answer_vec = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)
        self.answer_matrix = self.answer_vec.fit_transform([normalise(re.sub(r"\*", "", f["answer"]))
                                                            for f in self.faq])

    # ---------------------------------------------------------------- validation
    @staticmethod
    def validate(text) -> str | None:
        """Return an error message for invalid input, or None if the input is OK."""
        if text is None or not str(text).strip():
            return "Please type a question first. For example: *When is electricity demand highest?*"
        t = str(text).strip()
        if len(t) > MAX_CHARS:
            return f"That's {len(t)} characters. Please keep your question under {MAX_CHARS} characters."
        letters = re.findall(r"[A-Za-z]", t)
        if re.search(r"[぀-ヿ一-鿿]", t) and len(letters) < 3:
            return "Sorry, I can only understand questions in English for now (日本語は未対応です)."
        if len(letters) < 3:
            return "That doesn't look like a question. Please use words, e.g. *What was the record demand?*"
        if len(set(t.lower().replace(" ", ""))) <= 2:
            return "That doesn't look like a question. Please try again with a full sentence."
        return None

    # ---------------------------------------------------------------- dates
    def find_date(self, text: str):
        t = text.lower()
        m = re.search(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", t)
        if m:
            y, mo, d = map(int, m.groups())
        else:
            mon = "|".join(sorted(MONTHS, key=len, reverse=True))
            m1 = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({mon})\.?,?\s+(20\d{{2}})\b", t)
            m2 = re.search(rf"\b({mon})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d{{2}})\b", t)
            if m1:
                d, mo, y = int(m1.group(1)), MONTHS[m1.group(2)], int(m1.group(3))
            elif m2:
                mo, d, y = MONTHS[m2.group(1)], int(m2.group(2)), int(m2.group(3))
            else:
                return None
        try:
            return pd.Timestamp(year=y, month=mo, day=d)
        except ValueError:
            return "invalid"

    def answer_date(self, day) -> Result:
        if day == "invalid":
            return Result("invalid", "That date doesn't exist. Please check the day and month.")
        if self.daily is None:
            return Result("unsure", "The daily data file is missing, so I can't look up dates.")
        first, last = self.daily.index.min(), self.daily.index.max()
        if day < first or day > last:
            return Result("unsure", f"I only have data from **{first:%d %b %Y}** to **{last:%d %b %Y}**. "
                                    f"{day:%d %b %Y} is outside that range.", related=[])
        r = self.daily.loc[day]
        kind = "a workday" if bool(r["is_workday"]) else "a weekend/holiday"
        avg = self.daily["peak_demand_gw"].mean()
        diff = r["peak_demand_gw"] - avg
        msg = (f"**{day:%A, %d %B %Y}** ({kind}, {r['season'].lower()}):\n\n"
               f"- Peak demand: **{r['peak_demand_gw']:.1f} GW** at {int(r['peak_hour']):02d}:00 "
               f"({abs(diff):.1f} GW {'above' if diff >= 0 else 'below'} the average daily peak of {avg:.1f} GW)\n"
               f"- Lowest demand: **{r['low_demand_gw']:.1f} GW**, average {r['mean_demand_gw']:.1f} GW\n"
               f"- Tokyo temperature: **{r['min_temp_c']:.1f} – {r['max_temp_c']:.1f} °C** "
               f"(mean {r['mean_temp_c']:.1f} °C)")
        return Result("date", msg, confidence=1.0, matched_question="Date lookup in the real data",
                      category="Data lookup")

    # ---------------------------------------------------------------- retrieval
    def scores(self, text: str, return_word: bool = False):
        """Similarity of the question to every FAQ entry (best-matching phrasing per entry)."""
        q = normalise(text)
        q_words = re.sub(r"\b\d+\b", " ", q)          # numbers alone (e.g. "2022") must not create a match
        word = cosine_similarity(self.word_vec.transform([q_words]), self.word_matrix)[0]
        char = cosine_similarity(self.char_vec.transform([q]), self.char_matrix)[0]
        per_word, per_char = np.zeros(len(self.faq)), np.zeros(len(self.faq))
        np.maximum.at(per_word, self.owner, word)
        np.maximum.at(per_char, self.owner, char)
        per_answer = cosine_similarity(self.answer_vec.transform([q_words]), self.answer_matrix)[0]
        total = WEIGHTS["word"] * per_word + WEIGHTS["char"] * per_char + WEIGHTS["answer"] * per_answer
        return (total, per_word.max()) if return_word else total

    def ask(self, text) -> Result:
        error = self.validate(text)
        if error:
            return Result("invalid", error)
        text = str(text).strip()
        day = self.find_date(text)
        if day is not None:
            return self.answer_date(day)
        s, word_best = self.scores(text, return_word=True)
        order = np.argsort(s)[::-1]
        best = order[0]
        related = [self.faq[i]["question"] for i in order[1:4] if s[i] >= self.threshold * 0.6]
        in_domain = bool(DOMAIN_TERMS & set(normalise(text).split()))
        if s[best] < self.threshold or word_best < DOMAIN_GATE or not in_domain:
            suggestions = [self.faq[i]["question"] for i in order[:3]]
            return Result("unsure", "Sorry, I'm not sure I understood. I can answer questions about Tokyo "
                                    "electricity demand, weather, solar power and this project's data. "
                                    "Did you mean one of these?",
                          confidence=float(s[best]), related=suggestions)
        item = self.faq[best]
        return Result("answer", item["answer"], confidence=float(s[best]), matched_question=item["question"],
                      category=item["category"], related=related)


if __name__ == "__main__":
    bot = FAQAssistant()
    for q in ["When is power usage highest during the day?", "what was the recrod peak?", "", "???",
              "What was demand on 2024-08-05?", "How hot was it on 5 August 2024?", "What's the capital of France?",
              "電気の需要は？", "demand on 2030-01-01"]:
        r = bot.ask(q)
        print(f"\nQ: {q!r}\n-> {r.status} ({r.confidence:.2f}) {r.matched_question}\n   {r.message[:160]}")
