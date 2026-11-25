"""
Evaluate the question-answering model.

Two question sets, both written separately from faq.json:
  * dev_questions.csv         (72) - used while improving the model (so its score is optimistic)
  * final_test_questions.csv  (42) - held out; used only to compare versions
Each set has electricity questions with the expected answer id, plus off-topic questions (expected "none")
that the assistant should refuse.

    python evaluate.py

Metrics
  * Top-1 accuracy  - electricity questions answered with the right answer
  * Top-3 accuracy  - the right answer is among the 3 best matches (the app shows these as suggestions)
  * Rejection rate  - off-topic questions correctly refused
  * Overall         - all questions handled correctly
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qa_engine import FAQAssistant

HERE = Path(__file__).resolve().parent

# Held-out results of earlier versions, measured with the same final_test_questions.csv (see README).
HISTORY = {
    "v1: TF-IDF words + characters": {"top1": 81.8, "reject": 55.6, "overall": 76.2},
    "v2: + synonyms, answer text, tuned on dev": {"top1": 75.8, "reject": 66.7, "overall": 73.8},
}


def evaluate(bot: FAQAssistant, path: Path):
    ids = {f["question"]: f["id"] for f in bot.faq}
    all_ids = [f["id"] for f in bot.faq]
    t = pd.read_csv(path)
    rows = []
    for q, expected in zip(t["question"], t["expected_id"]):
        r = bot.ask(q)
        s = bot.scores(q)
        top3 = [all_ids[i] for i in np.argsort(s)[::-1][:3]]
        predicted = ids.get(r.matched_question, "rejected") if r.status == "answer" else "rejected"
        if expected == "none":
            result = "correctly rejected" if predicted == "rejected" else "wrongly answered"
        else:
            result = "correct" if predicted == expected else ("wrongly rejected" if predicted == "rejected"
                                                              else "wrong answer")
        rows.append({"question": q, "expected_id": expected, "predicted_id": predicted,
                     "confidence": round(r.confidence, 3), "in_top3": expected in top3, "result": result})
    res = pd.DataFrame(rows)
    ind, off = res[res["expected_id"] != "none"], res[res["expected_id"] == "none"]
    summary = {"questions": len(res), "electricity_questions": len(ind), "off_topic_questions": len(off),
               "top1_accuracy_pct": round((ind["result"] == "correct").mean() * 100, 1),
               "top3_accuracy_pct": round(ind["in_top3"].mean() * 100, 1),
               "off_topic_rejection_pct": round((off["result"] == "correctly rejected").mean() * 100, 1),
               "overall_accuracy_pct": round(res["result"].isin(["correct", "correctly rejected"]).mean() * 100, 1)}
    return summary, res


def threshold_sweep(path: Path):
    out = []
    for th in np.round(np.arange(0.10, 0.62, 0.04), 2):
        s, _ = evaluate(FAQAssistant(threshold=th), path)
        out.append({"threshold": th, **s})
    return pd.DataFrame(out)


def charts(final_summary, sweep, threshold):
    BLUE, ORANGE, INK, INK_2, GRID, SURFACE, NEUTRAL = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", \
        "#fcfcfb", "#c9c8c2"

    def style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, length=0, labelsize=8.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    versions = {**HISTORY, "v3: + domain vocabulary check (final)": {
        "top1": final_summary["top1_accuracy_pct"], "reject": final_summary["off_topic_rejection_pct"],
        "overall": final_summary["overall_accuracy_pct"]}}
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    style(ax)
    metrics = [("top1", "Electricity questions\nanswered correctly"), ("reject", "Off-topic questions\nrejected"),
               ("overall", "Overall accuracy")]
    x = np.arange(len(metrics))
    colors = [NEUTRAL, BLUE, ORANGE]
    for k, (name, v) in enumerate(versions.items()):
        vals = [v[m] for m, _ in metrics]
        ax.bar(x + (k - 1) * 0.26, vals, width=0.25, color=colors[k], label=name)
        for xi, val in zip(x, vals):
            ax.text(xi + (k - 1) * 0.26, val + 1.2, f"{val:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of held-out test questions", color=INK_2, fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), frameon=False, fontsize=8.5, labelcolor=INK, ncol=3)
    ax.set_title("Held-out test: the final version refuses off-topic questions far better", loc="left",
                 fontsize=12, fontweight="bold", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(HERE / "fig_version_comparison.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    style(ax)
    for col, color, name in [("top1_accuracy_pct", BLUE, "Electricity questions answered correctly"),
                             ("off_topic_rejection_pct", ORANGE, "Off-topic questions rejected"),
                             ("overall_accuracy_pct", INK, "Overall accuracy")]:
        ax.plot(sweep["threshold"], sweep[col], color=color, linewidth=2, label=name)
    ax.axvline(threshold, color=INK_2, linewidth=1, linestyle="--")
    ax.text(threshold + 0.006, 8, f"chosen threshold {threshold:.2f}", fontsize=8.5, color=INK_2)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Confidence threshold", color=INK_2, fontsize=9)
    ax.set_ylabel("% of development questions", color=INK_2, fontsize=9)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax.set_title("Choosing the confidence threshold (development set)", loc="left", fontsize=12,
                 fontweight="bold", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(HERE / "fig_threshold.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    bot = FAQAssistant()
    dev_summary, dev_res = evaluate(bot, HERE / "dev_questions.csv")
    test_summary, test_res = evaluate(bot, HERE / "final_test_questions.csv")
    sweep = threshold_sweep(HERE / "dev_questions.csv")
    pd.concat([dev_res.assign(set="dev"), test_res.assign(set="final_test")]).to_csv(
        HERE / "evaluation_results.csv", index=False)
    summary = {"threshold": bot.threshold, "knowledge_base_entries": len(bot.faq),
               "knowledge_base_phrasings": len(bot.texts), "development_set": dev_summary,
               "final_test_set": test_summary, "earlier_versions_on_final_test": HISTORY}
    (HERE / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    charts(test_summary, sweep, bot.threshold)
    return summary, dev_res, test_res, sweep


if __name__ == "__main__":
    summary, dev_res, test_res, sweep = main()
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
    print(json.dumps(summary, indent=2))
    print("\nMistakes on the final test set:")
    print(test_res[~test_res["result"].isin(["correct", "correctly rejected"])].to_string(index=False))
