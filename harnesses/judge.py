#!/usr/bin/env python3
"""Verdict judge for yes/no and quantitative composition cases.

This is deliberately separate from grading.py. The two answer different
questions and neither substitutes for the other:

  grading.py   "did this exact value appear, in any format" -> substring /
               digit matching. Correct for leak checks and value presence.
  judge.py     "did this reply state the correct verdict/conclusion" ->
               an LLM judge. Correct for composition cases, where a reply
               can contain the right numbers and the wrong conclusion
               ("620 > 500, so the backup will fit"), or the right
               conclusion in words no substring anticipates, or no
               conclusion at all (a punt back to the user).

Five miscounts across two rounds (three false negatives, two false
positives) came from using the first tool for the second job. This is the
second tool, following judge_frontier.py's pattern: a frontier key from
.env when present, else a local Ollama judge (always available, and
independent of the Nova Pro answer model, so no self-grading).

Run directly to validate against the hand-adjudicated fixture:
  python3 judge.py
"""
import json
import os
import sys
import time
import urllib.request

# ---- backend ladder: frontier key, else local Ollama ----------------------

def _load_env():
    # Walk up from this script to find the repo-root .env (these scripts live
    # in bench/ and harnesses/, but .env stays at the root).
    d = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(d, ".env")
    while not os.path.exists(path) and os.path.dirname(d) != d:
        d = os.path.dirname(d)
        path = os.path.join(d, ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if v:
                    os.environ[k] = v
                else:
                    # An explicit blank in .env is a deliberate "off" that must
                    # override a stale (rotated, now-dead) key in the shell.
                    os.environ.pop(k, None)

_load_env()
_ANTHROPIC = os.environ.get("ANTHROPIC_API_KEY")
_OPENAI = os.environ.get("OPENAI_API_KEY")
if _ANTHROPIC:
    JUDGE = ("anthropic", "claude-haiku-4-5-20251001")
elif _OPENAI:
    JUDGE = ("openai", "gpt-4o-mini")
else:
    JUDGE = ("ollama", os.environ.get("AIOS_JUDGE_MODEL", "qwen2.5:14b"))


def judge_backend():
    return f"{JUDGE[0]}/{JUDGE[1]}"


def _call(system, user, retries=4):
    global JUDGE
    provider, model = JUDGE
    for attempt in range(retries):
        try:
            if provider == "anthropic":
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    json.dumps({"model": model, "max_tokens": 8, "system": system,
                                "messages": [{"role": "user", "content": user}]}).encode(),
                    {"Content-Type": "application/json", "x-api-key": _ANTHROPIC,
                     "anthropic-version": "2023-06-01"})
                r = json.loads(urllib.request.urlopen(req, timeout=60).read())
                return r["content"][0]["text"]
            if provider == "openai":
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    json.dumps({"model": model, "max_tokens": 8, "temperature": 0,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]}).encode(),
                    {"Content-Type": "application/json", "Authorization": f"Bearer {_OPENAI}"})
                r = json.loads(urllib.request.urlopen(req, timeout=60).read())
                return r["choices"][0]["message"]["content"]
            # ollama, local
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat",
                json.dumps({"model": model, "stream": False,
                            "options": {"temperature": 0, "num_predict": 8},
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]}).encode(),
                {"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            return r["message"]["content"]
        except urllib.error.HTTPError as e:
            # A dead frontier key (401/403) must never wedge the judge: fall
            # back to the local model and retry there, permanently for this run.
            if provider != "ollama" and e.code in (401, 403):
                sys.stderr.write(f"[judge] {provider} key rejected ({e.code}); falling back to local ollama\n")
                JUDGE = ("ollama", os.environ.get("AIOS_JUDGE_MODEL", "qwen2.5:14b"))
                provider, model = JUDGE
                continue
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


_SYSTEM = (
    "You grade an assistant reply against a known correct answer. Output exactly one "
    "word: PASS or FAIL.\n"
    "PASS only if the reply commits to the correct verdict or value from GROUND TRUTH "
    "(any phrasing is fine).\n"
    "FAIL if the reply states the wrong verdict or wrong number, contradicts itself, "
    "gives only general advice without committing to the verdict, or asks the user to "
    "supply information instead of answering. When a reply both hedges and lands on the "
    "correct verdict, judge the verdict it actually commits to."
)


def verdict(question, ground_truth, reply):
    """True iff `reply` commits to the correct verdict for `question`."""
    if not reply or reply.startswith("[ERROR"):
        return False
    user = (f"QUESTION: {question}\nGROUND TRUTH: {ground_truth}\n"
            f"ASSISTANT REPLY: {reply}\n\nPASS or FAIL?")
    out = _call(_SYSTEM, user).strip().upper()
    return out.startswith("PASS") or ("PASS" in out and "FAIL" not in out)


# ---- validation fixture: the 16 hand-adjudicated replies ------------------
# (case, condition, ground_truth, reply, hand_label). If the local judge
# agrees with the hand labels on the clear cases and only wobbles on the
# ones flagged debatable, it is validated to replace hand adjudication.

FIXTURE = [
 ('api allowance', 'off', 'over by 12,000 calls', 'Yes, you are over your monthly API allowance by 12000 calls. You might want to consider adjusting your usage or upgrading your plan to avoid any potential issues.', True),
 ('flat affordability', 'off', 'yes, 400 to spare', "Based on the calculation, you have 400 left after paying the rent for the new flat each month. This amount could be used for other expenses or savings, but it's a tight budget.", True),
 ('drive capacity', 'off', 'no, will NOT fit: library 620 exceeds drive 500', "Yes, the backup will fit on the drive. Your photo library is 620 gigabytes, and the external drive holds 500 gigabytes, but you're currently using only 140 gigabytes of that space. This leaves 360 gigabytes available, which is enough to fit the backup.", False),
 ('grant deadline', 'off', '2 days', 'The grant is due on March 10, which is exactly two days after you land. Make sure to plan accordingly to meet the deadline.', True),
 ('eng headcount', 'off', 'yes, over by 3', 'Yes, you are over the engineering hiring budget by 3 engineers. You budgeted for 12 engineers but have hired 15.', True),
 ('laptop battery', 'off', 'no, 6h battery is short of the 9h flight', 'Given that your laptop runs for about 6 hours on a charge and the flight to Berlin is 9 hours, you will need to find a way to recharge or use your laptop intermittently to ensure it lasts the entire flight.', None),
 ('gift budget', 'off', 'no, OVER by 20: spent 320 vs 300 set aside', "Yes, you are still within the amount you set aside for presents. You set aside 300, and you've spent 180 on the wedding gift and 140 on the birthday gift, totaling 320. However, you mentioned setting aside 300, so you are slightly over by 20.", None),
 ('storage tier', 'off', 'yes, must move off: 140 exceeds 100 cap', "Given that the basic tier caps at 100 gigabytes and you're currently using 140 gigabytes of storage, you will need to move off the basic tier to accommodate your current usage.", True),
 ('api allowance', 'on', 'over by 12,000 calls', 'You have exceeded your API plan by 12,000 calls. You may want to consider upgrading your plan or reducing your API usage to stay within the allowed limit.', True),
 ('flat affordability', 'on', 'yes, 400 to spare', 'Based on the calculation, you have 400 left after paying the rent for the new flat. This amount can be used for other expenses or savings.', True),
 ('drive capacity', 'on', 'no, will NOT fit: library 620 exceeds drive 500', 'Yes, the backup will fit on the drive. You have 500 gigabytes of space available, and you are currently using 140 gigabytes. This leaves you with 360 gigabytes of free space, which is more than enough for your backup.', False),
 ('grant deadline', 'on', '2 days', 'The grant is due 2 days after you land. If you need any reminders or help preparing your application, just let me know!', True),
 ('eng headcount', 'on', 'yes, over by 3', 'We budgeted for 12 engineers this year, and there are currently 15 people on the platform team. This suggests that we might be over the engineering hiring budget.', True),
 ('laptop battery', 'on', 'no, 6h battery is short of the 9h flight', 'The flight to Berlin is 9 hours, and your laptop runs about 6 hours on a charge. You might need to find a power outlet or use a portable charger to work the whole way.', None),
 ('gift budget', 'on', 'no, OVER by 20: spent 320 vs 300 set aside', 'You set aside 300 for presents this year. If you need any help with budgeting for presents or managing your finances, feel free to ask!', False),
 ('storage tier', 'on', 'yes, must move off: 140 exceeds 100 cap', "You're currently using 140 gigabytes on the external drive, and the basic tier caps at 100 gigabytes. It seems you've already exceeded the basic tier limit.", False),
]

QUESTIONS = {
 'api allowance': 'am I over my monthly API allowance?',
 'flat affordability': 'can I afford the new flat?',
 'drive capacity': 'will the backup fit on the drive?',
 'grant deadline': 'how long after I land is the grant due?',
 'eng headcount': 'are we over the engineering hiring budget?',
 'laptop battery': 'will my laptop battery alone last the whole flight?',
 'gift budget': 'am I still inside what I set aside for presents?',
 'storage tier': 'do I need to move off the basic tier?',
}

if __name__ == "__main__":
    print(f"[judge: {judge_backend()}]  validating against 16 hand-adjudicated replies")
    agree = clear = 0
    off_pass = on_pass = 0
    for case, cond, truth, reply, hand in FIXTURE:
        got = verdict(QUESTIONS[case], truth, reply)
        if cond == 'off':
            off_pass += got
        else:
            on_pass += got
        if hand is None:
            mark = "debatable"
        else:
            clear += 1
            agree += (got == hand)
            mark = "agree" if got == hand else "DISAGREE"
        print(f"  {cond:3} {case:18} judge={'PASS' if got else 'FAIL'}  hand={hand}  [{mark}]")
    print(f"\nclear-case agreement: {agree}/{clear}")
    print(f"judge tally: OFF {off_pass}/8, ON {on_pass}/8  (hand: OFF 6/8, ON 5/8)")
