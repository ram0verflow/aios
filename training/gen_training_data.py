#!/usr/bin/env python3
"""Generate page-fault fine-tuning data from LoCoMo convs 1-9 (conv 0 = eval holdout).

Three behaviors, one protocol:
  ANSWER — evidence loaded in the memory block          -> concise gold answer
  FAULT  — evidence deliberately withheld               -> CONTEXT_NEEDED: <topic>
  REFUSE — plausible neighbor context loaded (entity    -> CONTEXT_NEEDED: <topic>
           swap traps from LoCoMo adversarial rows)

The system prompt replicates the Rust kernel's SYSTEM_TEMPLATE so the training
distribution matches deployment. Output: MLX chat JSONL (train/valid).
"""
import json
import random
import re

random.seed(7)

LOCOMO = "data/locomo10.json"
OUT_TRAIN = "ft_data/train.jsonl"
OUT_VALID = "ft_data/valid.jsonl"

# Mirror of kernel.rs SYSTEM_TEMPLATE (keep in sync).
SYSTEM_TEMPLATE = """You are a personal AI assistant with persistent, OS-managed memory.
Below is the memory currently paged into your context. Answer using ONLY this context.

RULES:
- Answer with the shortest phrase that fully answers the question — no preamble,
  no "Based on the context", no restating the question. "7 May 2023" beats
  "According to the conversation, it was on 7 May 2023."
- If the answer is in your loaded context, answer directly and concisely.
- If the user asks about something NOT in your loaded context, respond with EXACTLY:
  CONTEXT_NEEDED: <topic>
  and nothing else. Do NOT guess, infer, or invent facts that are not present.
- Memory blocks are namespaced (e.g. /social, /workspace). Do not mix rules across namespaces.
- Messages carry [timestamp] prefixes. For "when" questions, derive the date from those
  timestamps. Relative phrases inside a message ("last week", "yesterday") are relative
  to THAT message's timestamp — resolve them (e.g. "last week" said on 9 June 2023 means
  the week before 9 June 2023). Answer with the resolved date.
- A [TIME NOTES] block may follow the messages with relative dates ALREADY RESOLVED
  by the memory system. Trust those resolutions verbatim for "when" questions.

--- LOADED MEMORY ---
{context}
--- END MEMORY ---"""

STOP = set("a an the is it in on at to for of and or but not with this that from by be was were are am have has had do does did will would could should can may might i you he she we they my your his her our their me him us them its what which who whom how when where why if then so no yes about did".split())


def topic_of(question: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in STOP and len(w) > 2]
    return " ".join(words[:4]) if words else "unknown"


def load_conversations():
    data = json.load(open(LOCOMO))
    convs = []
    for conv in data[1:10]:  # conv 0 held out for eval
        c = conv["conversation"]
        turns = {}      # dia_id -> (speaker, text, session_time)
        order = []      # dia_ids in chronological order
        for k in sorted(c.keys()):
            m = re.match(r"session_(\d+)$", k)
            if not m:
                continue
            ts = c.get(f"session_{m.group(1)}_date_time", "")
            for t in c[k] or []:
                did = t.get("dia_id")
                if not did:
                    continue
                turns[did] = (t["speaker"].lower(), t["text"], ts)
                order.append(did)
        convs.append({"qa": conv["qa"], "turns": turns, "order": order})
    return convs


def render(dia_ids, turns):
    lines = []
    for did in dia_ids:
        sp, tx, ts = turns[did]
        lines.append(f"[{ts}] {sp}: {tx}" if ts else f"{sp}: {tx}")
    return "\n".join(lines)


def build_context(conv, include, n_total=26):
    """Chronological context of ~n_total msgs containing `include` dia_ids."""
    order, turns = conv["order"], conv["turns"]
    chosen = set(include)
    # neighbors of evidence for conversational coherence
    for did in include:
        if did in order:
            i = order.index(did)
            for j in (i - 1, i + 1):
                if 0 <= j < len(order):
                    chosen.add(order[j])
    pool = [d for d in order if d not in chosen]
    random.shuffle(pool)
    for d in pool[: max(0, n_total - len(chosen))]:
        chosen.add(d)
    ids = [d for d in order if d in chosen]
    return render(ids, turns)


def build_context_excluding(conv, exclude, n_total=26):
    """Context that avoids `exclude` dia_ids and their neighbors entirely."""
    order, turns = conv["order"], conv["turns"]
    banned = set(exclude)
    for did in exclude:
        if did in order:
            i = order.index(did)
            for j in (i - 1, i + 1):
                if 0 <= j < len(order):
                    banned.add(order[j])
    pool = [d for d in order if d not in banned]
    random.shuffle(pool)
    keep = set(pool[:n_total])
    ids = [d for d in order if d in keep]
    return render(ids, turns)


def example(question, context, target, topic):
    ctx = f"[MEMORY_BLOCK: /social/{topic.replace(' ', '_')}]\n{context}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_TEMPLATE.replace("{context}", ctx)},
            {"role": "user", "content": question},
            {"role": "assistant", "content": target},
        ]
    }


def context_dia_ids(conv, include, n_total=26):
    """Same selection logic as build_context but returns the dia_id set."""
    # build_context renders; for boundary twins we need to know WHICH ids
    # landed in the context, so we recompute the selection deterministically.
    order, turns = conv["order"], conv["turns"]
    chosen = set(include)
    for did in include:
        if did in order:
            i = order.index(did)
            for j in (i - 1, i + 1):
                if 0 <= j < len(order):
                    chosen.add(order[j])
    return chosen


def main():
    convs = load_conversations()
    examples = []
    n_ans = n_fault = n_refuse = n_twin = n_multihop = 0

    for conv in convs:
        answerable = [q for q in conv["qa"] if "answer" in q]
        for i, q in enumerate(conv["qa"]):
            question = q.get("question", "")
            evidence = [e for e in (q.get("evidence") or []) if e in conv["turns"]]
            topic = topic_of(question)

            if "answer" in q:
                gold = q["answer"] if isinstance(q["answer"], str) else str(q["answer"])
                if evidence:
                    ctx = build_context(conv, evidence)
                    examples.append(example(question, ctx, gold, topic))
                    n_ans += 1
                    # Round 2: multi-hop answers (2+ evidence msgs) upweighted 2x —
                    # round 1 taught "not verbatim in one message -> fault" and
                    # multi-hop cratered 69%->34%. Synthesis needs its own weight.
                    if len(evidence) >= 2:
                        examples.append(example(question, build_context(conv, evidence), gold, topic))
                        n_multihop += 1
                    # Round 2: faults cut from 1-in-2 to 1-in-7 (over-faulting fix).
                    if i % 7 == 0:
                        ctx = build_context_excluding(conv, evidence)
                        examples.append(example(question, ctx, f"CONTEXT_NEEDED: {topic}", topic))
                        n_fault += 1
            elif "adversarial_answer" in q:
                # Trap content IS loaded; correct move: fault.
                ctx = build_context(conv, evidence) if evidence else build_context_excluding(conv, [])
                examples.append(example(question, ctx, f"CONTEXT_NEEDED: {topic}", topic))
                n_refuse += 1
                # Boundary twin: the SAME trap context, asked something it DOES
                # answer, teaches the boundary, not blanket refusal.
                if evidence:
                    in_ctx = context_dia_ids(conv, evidence)
                    for aq in answerable:
                        aev = [e for e in (aq.get("evidence") or []) if e in conv["turns"]]
                        if aev and all(e in in_ctx for e in aev):
                            agold = aq["answer"] if isinstance(aq["answer"], str) else str(aq["answer"])
                            examples.append(example(aq["question"], ctx, agold, topic_of(aq["question"])))
                            n_twin += 1
                            break

    random.shuffle(examples)
    split = max(1, len(examples) // 20)
    valid, train = examples[:split], examples[split:]

    import os
    os.makedirs("ft_data", exist_ok=True)
    with open(OUT_TRAIN, "w") as f:
        for e in train:
            f.write(json.dumps(e) + "\n")
    with open(OUT_VALID, "w") as f:
        for e in valid:
            f.write(json.dumps(e) + "\n")

    print(f"answer={n_ans} (+{n_multihop} multihop dups, +{n_twin} boundary twins) "
          f"fault={n_fault} refuse={n_refuse} total={len(examples)}")
    answerish = n_ans + n_multihop + n_twin
    print(f"mix: answer {100*answerish//len(examples)}% | fault {100*n_fault//len(examples)}% | refuse {100*n_refuse//len(examples)}%")
    print(f"train={len(train)} valid={len(valid)}")
    lens = [len(json.dumps(e)) // 4 for e in examples]
    print(f"approx tokens/example: mean={sum(lens)//len(lens)} max={max(lens)}")


if __name__ == "__main__":
    main()
