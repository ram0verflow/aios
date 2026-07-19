#!/usr/bin/env python3
"""Compaction stress test against a live aios daemon.

Plants facts, buries them under dozens of distractor turns on a deliberately
tiny session window (so the eviction/demotion machinery has to churn), then
asks for every fact back and grades the answers. State is whatever daemon it
points at; run the daemon with AIOS_HOME=/tmp/aios-stress to keep your real
memory out of it.

  AIOS_HOME=/tmp/aios-stress ./target/release/aios-daemon --port 4311 &
  python3 stress_daemon.py 4311
"""

import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 4310
BASE = f"http://127.0.0.1:{PORT}"

FACTS = [
    ("My dentist appointment is on October 14th.", "when is my dentist appointment?", "october 14"),
    ("My cat is called Biscuit.", "what is my cat called?", "biscuit"),
    ("I parked the car on level 4B of the airport garage.", "where did I park at the airport?", "4b"),
    ("My wifi password hint is: the street I grew up on.", "what is my wifi password hint?", "street"),
    ("I lent Rohan my copy of SICP.", "who did I lend SICP to?", "rohan"),
    ("The production database runs Postgres 16.", "which Postgres version does production run?", "16"),
    ("My locker combination at the gym is 7-31-19.", "what is my gym locker combination?", "7-31-19"),
    ("Mum's birthday dinner is booked at Trattoria Nonna.", "where is mum's birthday dinner booked?", "nonna"),
    ("My flight lands in Lisbon at 9:40 in the morning.", "what time does my flight land in Lisbon?", "9:40"),
    ("The API rate limit we agreed on is 120 requests per minute.", "what API rate limit did we agree on?", "120"),
]

DISTRACTORS = [
    "What do you think makes a good operating system design?",
    "I've been listening to a lot of jazz lately, any thoughts on Coltrane?",
    "Explain how BM25 scoring works in a sentence or two.",
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


def main():
    t_start = time.time()
    status = get("/v1/status")
    print(f"daemon on :{PORT} | model {status['provider']}/{status['model']} "
          f"| memory brain {status['local_model']}")

    # A window small enough that compaction is constant.
    put_settings({"window_budget": 500})
    print("window budget forced to 500 tokens\n")

    print(f"— planting {len(FACTS)} facts")
    for i, (fact, _, _) in enumerate(FACTS):
        reply, _ = turn(fact)
        print(f"  [{i+1:2}] {fact[:52]:52} -> {reply[:44]!r}")

    print(f"\n— burying them under {len(DISTRACTORS)} distractor turns")
    max_used = 0
    for i, d in enumerate(DISTRACTORS):
        turn(d)
        used, budget, evictions, level = pressure()
        max_used = max(max_used, used)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2}/{len(DISTRACTORS)}] window {used}/{budget} ({level}), "
                  f"{evictions} demotions so far")

    print("\n— recall")
    hits, results = 0, []
    for i, (fact, question, needle) in enumerate(FACTS):
        reply, done = turn(question)
        insp = (done or {}).get("inspector", {})
        ok = needle.lower() in reply.lower()
        hits += ok
        results.append({"fact": fact, "question": question, "needle": needle,
                        "reply": reply, "ok": ok,
                        "loaded": insp.get("loaded"),
                        "retrieval_ms": insp.get("retrieval_ms"),
                        "faulted": insp.get("faulted")})
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {question[:44]:44} -> {reply[:60]!r}")

    used, budget, evictions, level = pressure()
    store = get("/v1/status")["counters"]["store"]
    elapsed = time.time() - t_start
    print(f"\n=== recall {hits}/{len(FACTS)} "
          f"| window peaked at {max_used}/{budget} tokens, never exceeded "
          f"| {evictions} messages demoted to archive "
          f"| store now {store['branches']} topics / {store['details']} facts "
          f"| {len(FACTS)*2 + len(DISTRACTORS)} turns in {elapsed/60:.1f} min ===")

    with open("/tmp/aios_stress_report.json", "w") as f:
        json.dump({"recall": hits, "total": len(FACTS), "max_window_used": max_used,
                   "budget": budget, "evictions": evictions, "store": store,
                   "elapsed_s": elapsed, "results": results}, f, indent=2)
    print("full report: /tmp/aios_stress_report.json")


if __name__ == "__main__":
    main()
