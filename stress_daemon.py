#!/usr/bin/env python3
"""Compaction stress test against a live aios daemon.

Measures three things separately, because they are different claims:

  1. retrieval survives window churn: plant facts, force a tiny session
     window so every fact gets evicted, bury them under distractors, ask
     for them back. Attribution is structural: in the daemon's flow the
     store never enters the prompt (identity aside), so recall rides the
     driver's message index.
  2. write-back capture: how many planted facts actually landed in the
     store as facts. This is the memory-formation path, graded against
     /v1/memory/browse at the end.
  3. discipline: two facts get contradicted/updated mid-run (the answer
     must be the new value, not the old), and two probes ask for things
     never said (the answer must not be a confident wrong fact).

State is whatever daemon it points at; run with AIOS_HOME somewhere
disposable to keep real memory out of it:

  AIOS_HOME=/tmp/aios-stress ./target/release/aios-daemon --port 4311 &
  python3 stress_daemon.py 4311
"""

import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 4310
BASE = f"http://127.0.0.1:{PORT}"

# (planted sentence, recall question, expected needle, forbidden stale
#  needle, grammatical form). The form tag exists to see whether capture
#  misses cluster by phrasing rather than randomly.
FACTS = [
    ("My dentist appointment is on October 14th.", "when is my dentist appointment?", "october 21", "october 14", "possessive+update"),
    ("My cat is called Biscuit.", "what is my cat called?", "biscuit", None, "possessive"),
    ("I parked the car on level 4B of the airport garage.", "where did I park at the airport?", "4b", None, "event"),
    ("My wifi password hint is: the street I grew up on.", "what is my wifi password hint?", "street", None, "possessive"),
    ("I lent Rohan my copy of SICP.", "who did I lend SICP to?", "rohan", None, "event+third-party"),
    ("The production database runs Postgres 16.", "which Postgres version does production run?", "16", None, "third-person fact"),
    ("My locker combination at the gym is 7-31-19.", "what is my gym locker combination?", "7-31-19", None, "possessive"),
    ("Mum's birthday dinner is booked at Trattoria Nonna.", "where is mum's birthday dinner booked?", "nonna", None, "third-person fact"),
    ("My flight lands in Lisbon at 9:40 in the morning.", "what time does my flight land in Lisbon?", "9:40", None, "itinerary aside"),
    ("The API rate limit we agreed on is 120 requests per minute.", "what API rate limit did we agree on?", "90", "120", "agreement+update"),
]

# Injected mid-distractor: the versioning/dedup path has to handle these.
UPDATES = {
    10: "Change of plans: the dentist moved my appointment to October 21st.",
    20: "Correction on the API: we lowered the rate limit to 90 requests per minute.",
}

# Things never said. A confident wrong answer here is the real failure mode.
PROBES = [
    ("what is my locker combination at the pool?", "7-31-19"),
    ("when is my brother's wedding?", "october"),
]

DISTRACTORS = [
    "What do you think makes a good operating system design?",
    "I've been listening to a lot of jazz lately, any thoughts on Coltrane?",
    "Explain how keyword scoring works in a sentence or two.",
    "I'm thinking about repainting the study. Maybe a warm off-white.",
    "What's a good warmup routine before a run?",
    "Tell me something interesting about the history of Lisbon.",
    "How do you feel about tabs versus spaces?",
    "I had a great espresso this morning, tiny cafe near the station.",
    "What's the difference between a process and a thread?",
    "Recommend a novel for a long flight.",
    "Why do laptops throttle under sustained load?",
    "I saw a heron by the river today, huge thing.",
    "What makes sourdough different from regular bread?",
    "Summarize what a page fault is for a five year old.",
    "Do you prefer mornings or evenings? I'm a night person.",
    "What should I know before adopting a second cat?",
    "How does spaced repetition work?",
    "The gym was packed today, could barely get a bench.",
    "What's your take on keyboard shortcuts versus mice?",
    "Explain eventual consistency without jargon.",
    "I might switch my editor theme to something warmer.",
    "What causes jet lag exactly?",
    "Rust lifetimes finally clicked for me yesterday.",
    "What's a reasonable amount of RAM for a dev laptop in 2026?",
    "Tell me a short fact about the moon.",
    "My neighbour is learning the violin. Slowly.",
    "How do noise cancelling headphones work?",
    "What's the deal with kombucha?",
    "Describe the actor model in two sentences.",
    "I keep meaning to learn to juggle.",
]


def turn(text, timeout=180):
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/turn", data=body, headers={"Content-Type": "application/json"}
    )
    reply, done = "", None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev.get("t") == "done":
                done = ev
                reply = ev.get("reply", "")
    return reply, done


def get(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
        return json.load(r)


def put_settings(patch):
    body = json.dumps(patch).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/settings", data=body, method="PUT",
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=30).read()


def pressure():
    p = get("/v1/status")["pressure"]
    return p["used"], p["budget"], p["evictions"], p["level"]


def store_text():
    b = get("/v1/memory/browse")
    parts = [b["identity"]["current"]]
    for br in b["branches"]:
        parts.append(br["name"])
        parts.append(br["summary"]["current"])
        parts.extend(d["current"] for d in br["details"])
    return " ".join(parts).lower(), b


def main():
    t_start = time.time()
    status = get("/v1/status")
    print(f"daemon on :{PORT} | model {status['provider']}/{status['model']} "
          f"| memory brain {status['local_model']}")

    put_settings({"window_budget": 500})
    print("window budget forced to 500 tokens\n")

    print(f"— planting {len(FACTS)} facts")
    for i, (fact, _, _, _, _) in enumerate(FACTS):
        reply, _ = turn(fact)
        print(f"  [{i+1:2}] {fact[:52]:52} -> {reply[:44]!r}")

    print(f"\n— burying them under {len(DISTRACTORS)} distractor turns "
          f"(+{len(UPDATES)} contradicting updates)")
    max_used = 0
    for i, d in enumerate(DISTRACTORS):
        if i in UPDATES:
            up_reply, _ = turn(UPDATES[i])
            print(f"  [update @{i}] {UPDATES[i][:46]:46} -> {up_reply[:36]!r}")
        turn(d)
        used, budget, evictions, level = pressure()
        max_used = max(max_used, used)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1:2}/{len(DISTRACTORS)}] window {used}/{budget} ({level}), "
                  f"{evictions} demotions so far")

    print("\n— recall (updated facts must answer with the NEW value)")
    hits, stale, results = 0, 0, []
    for fact, question, needle, forbidden, form in FACTS:
        reply, done = turn(question)
        insp = (done or {}).get("inspector", {})
        low = reply.lower()
        ok = needle.lower() in low
        went_stale = bool(forbidden) and forbidden.lower() in low and not ok
        hits += ok
        stale += went_stale
        results.append({"fact": fact, "question": question, "needle": needle,
                        "form": form,
                        "reply": reply, "ok": ok, "stale": went_stale,
                        "loaded": insp.get("loaded"),
                        "retrieval_ms": insp.get("retrieval_ms"),
                        "faulted": insp.get("faulted")})
        mark = "PASS" if ok else ("STALE" if went_stale else "FAIL")
        print(f"  [{mark:5}] {question[:42]:42} -> {reply[:58]!r}")

    print("\n— probes (things never said; a confident wrong fact is the failure)")
    probe_ok, probe_results = 0, []
    for question, must_not in PROBES:
        reply, done = turn(question)
        insp = (done or {}).get("inspector", {})
        ok = must_not.lower() not in reply.lower()
        probe_ok += ok
        probe_results.append({"question": question, "must_not": must_not,
                              "reply": reply, "ok": ok,
                              "faulted": insp.get("faulted")})
        mark = "PASS" if ok else "LEAK"
        print(f"  [{mark:5}] {question[:42]:42} -> {reply[:58]!r}")

    # Write-back capture: which planted facts exist in the STORE at all,
    # and whether misses cluster by grammatical form.
    stext, browse = store_text()
    captured = 0
    print("\n— write-back capture by form")
    for _, _, needle, _, form in FACTS:
        got = needle.lower() in stext
        captured += got
        print(f"  [{'HIT ' if got else 'MISS'}] {form:20} {needle}")
    n_details = sum(len(b["details"]) for b in browse["branches"])

    used, budget, evictions, level = pressure()
    elapsed = time.time() - t_start
    n_turns = len(FACTS) * 2 + len(DISTRACTORS) + len(UPDATES) + len(PROBES)
    print(f"\n=== retrieval under churn: {hits}/{len(FACTS)} "
          f"({stale} answered stale) "
          f"| probes clean: {probe_ok}/{len(PROBES)} "
          f"| store capture: {captured}/{len(FACTS)} needles present "
          f"({len(browse['branches'])} topics, {n_details} details) "
          f"| window peak {max_used}/{budget}, {evictions} demotions "
          f"| {n_turns} turns in {elapsed/60:.1f} min ===")
    print("note: generation-time recall is served by the driver index plus "
          "identity; the store never enters the prompt. Capture measures the "
          "write-back path on its own.")

    with open("/tmp/aios_stress_report.json", "w") as f:
        json.dump({"recall": hits, "stale": stale, "total": len(FACTS),
                   "probes_ok": probe_ok, "probes": probe_results,
                   "store_capture": captured, "store_details": n_details,
                   "max_window_used": max_used, "budget": budget,
                   "evictions": evictions, "elapsed_s": elapsed,
                   "results": results}, f, indent=2)
    print("full report: /tmp/aios_stress_report.json")


if __name__ == "__main__":
    main()
