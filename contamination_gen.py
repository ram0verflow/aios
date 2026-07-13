import json, os, urllib.request

data = json.load(open("data/locomo10.json"))
qa = data[0]["qa"]
path = "fullbench/contamination_conv0.jsonl"
done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
out = open(path, "a")
n = 0
for q in qa:
    gold = q.get("answer")
    if gold is None:
        continue
    n += 1
    if n <= done:
        continue  # resume
    gold = gold if isinstance(gold, str) else str(gold)
    body = json.dumps({
        "model": "llama3.1:8b",
        "messages": [
            {"role": "system", "content": "Answer in the shortest phrase possible. If you do not know, reply exactly: I don't know."},
            {"role": "user", "content": q["question"]},
        ],
        "stream": False, "options": {"num_ctx": 1024, "num_predict": 60, "temperature": 0},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
    pred = json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"].strip()
    out.write(json.dumps({"qid": n, "cat": str(q.get("category","")), "adv": False,
                          "question": q["question"], "gold": gold, "pred": pred, "system": "no-memory"}) + "\n")
    out.flush()
print(f"contamination baseline complete: {n} predictions")
