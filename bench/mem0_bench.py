#!/usr/bin/env python3
"""Mem0-OSS head-to-head on LoCoMo, same protocol as the AIOS sweep.

Memory layer: mem0 (OSS), fully local — Ollama for its extraction LLM and
embeddings, embedded vector store. Answer model: llama3.1:8b (same as AIOS
sweep). Output: same JSONL schema, judged by the same frontier judge.

Usage: python3 mem0_bench.py [conv_idx ...]   (default: all 10)
"""
import json
import os
import sys
import time
import urllib.request

from mem0 import Memory

LOCOMO = "data/locomo10.json"
OUT_DIR = "fullbench"

CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {"model": "llama3.1:8b", "temperature": 0, "ollama_base_url": "http://localhost:11434"},
    },
    "embedder": {
        "provider": "ollama",
        "config": {"model": "nomic-embed-text", "ollama_base_url": "http://localhost:11434"},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"path": "/tmp/mem0_qdrant", "collection_name": "locomo", "embedding_model_dims": 768},
    },
}

ANSWER_SYSTEM = (
    "You are a personal AI assistant with memory. Below are memories retrieved for the "
    "user's question. Answer ONLY from these memories, in the shortest phrase possible. "
    "If the memories don't contain the answer, say: I don't have that information."
)


def ollama_chat(system, user, n_predict=200):
    body = json.dumps({
        "model": "llama3.1:8b",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"num_ctx": 4096, "num_predict": n_predict, "temperature": 0},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["message"]["content"]


def conv_sessions(conv):
    c = conv["conversation"]
    nums = sorted(int(k.split("_")[1]) for k in c if k.startswith("session_") and "date" not in k)
    for sn in nums:
        ts = c.get(f"session_{sn}_date_time", "")
        turns = c.get(f"session_{sn}") or []
        msgs = [{"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"{t['speaker']}: {t['text']}"}
                for i, t in enumerate(turns) if t.get("text")]
        if msgs:
            yield sn, ts, msgs


def main():
    data = json.load(open(LOCOMO))
    args = sys.argv[1:]
    answers_only = "--answers-only" in args
    conv_ids = [int(a) for a in args if a.isdigit()] or list(range(10))
    os.makedirs(OUT_DIR, exist_ok=True)
    m = Memory.from_config(CONFIG)

    for ci in conv_ids:
        conv = data[ci]
        user_id = f"conv{ci}"
        t0 = time.time()

        if not answers_only:
            # --- Ingest: one add() per session, timestamped. ---
            n_sessions = 0
            for sn, ts, msgs in conv_sessions(conv):
                try:
                    m.add(msgs, user_id=user_id, metadata={"session": sn, "timestamp": ts})
                except Exception as e:
                    print(f"[conv{ci} s{sn}] add error: {e}", file=sys.stderr)
                n_sessions += 1
            print(f"[conv{ci}] ingested {n_sessions} sessions in {time.time()-t0:.0f}s", flush=True)

        # --- Answer all QA. ---
        out = open(f"{OUT_DIR}/mem0_conv{ci}.jsonl", "w")
        n = 0
        for q in conv["qa"]:
            question = q.get("question", "")
            is_adv = "adversarial_answer" in q
            gold = q.get("answer")
            if gold is None and not is_adv:
                continue
            gold = gold if isinstance(gold, str) else (json.dumps(gold) if gold is not None else "")
            try:
                hits = m.search(question, filters={"user_id": user_id}, limit=10)
                mems = hits.get("results", hits) if isinstance(hits, dict) else hits
                mem_text = "\n".join(f"- [{r.get('metadata',{}).get('timestamp','')}] {r.get('memory','')}"
                                     for r in mems)
            except Exception as e:
                mem_text = ""
                print(f"[conv{ci}] search error: {e}", file=sys.stderr)
            try:
                pred = ollama_chat(ANSWER_SYSTEM, f"Memories:\n{mem_text}\n\nQuestion: {question}")
            except Exception as e:
                pred = f"[ERROR {e}]"
            rec = {"qid": n + 1, "cat": str(q.get("category", "")), "adv": is_adv,
                   "question": question, "gold": gold, "pred": pred.strip(), "system": "mem0"}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n += 1
        out.close()
        print(f"[conv{ci}] answered {n} questions, total {time.time()-t0:.0f}s", flush=True)

    print("MEM0 BENCH COMPLETE")


if __name__ == "__main__":
    main()
