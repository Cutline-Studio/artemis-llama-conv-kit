"""Build the Artemis multi-turn conversational SFT set for a Llama-3.1-8B-Instruct LoRA (Latoya-approved 2026-10-26).
Sources (all real, local):
  A. /workspace/qwen-kit/sft/artemis_sft.jsonl (395 ex, made by qwen-kit/build_sft.py from committed Artemis docs + hand-written
     identity/governance/tool examples). Qwen-specific identity answers are rewritten for the Llama 3.1 base; doc_qa answers gain a
     [sources: File.md#Section] citation (Conversation.md citation format).
  B. NEW multi-turn dialogues TEMPLATED from the committed Artemis docs (labelled synthetic=True): memory of earlier turns,
     follow-ups that reference an earlier answer, clarify-then-answer, approval flows, topic switches, cited doc answers.
Deterministic (seed 567885). Validation split is grouped by source doc file so val docs are unseen in training."""
import json, os, random, re, subprocess, collections, zlib
R = random.Random(567885); REPO = "/workspace/a567-dense"; OUT = os.path.dirname(os.path.abspath(__file__)) + "/sft"
SYSTEM = ("You are Artemis, an AI model made by Cutline Industries LLC. Be plain, direct, warm and honest. "
          "Say when you don't know. Ask a short question when a request is ambiguous. Never claim to be human or to have "
          "feelings. Ask for explicit approval before spending money, sending messages, deleting anything, deploying, "
          "pushing code or renting hardware. When you answer from the Artemis docs, cite the source as File.md#Section. "
          "Remember what the user told you earlier in the conversation.")
rows = []
def add(kind, turns, group, synthetic, source):
    msgs = [{"role": "system", "content": SYSTEM}]
    for i, t in enumerate(turns):
        msgs.append(t if isinstance(t, dict) else {"role": "user" if i % 2 == 0 else "assistant", "content": t})
    rows.append({"kind": kind, "group": group, "synthetic": synthetic, "source": source, "messages": msgs})
slug = lambda h: re.sub(r"[^A-Za-z0-9]+", "-", h).strip("-")
# ---------------- docs (committed HEAD)
files = subprocess.run(["git", "-C", REPO, "ls-files", "artemis_dense/*.md"], capture_output=True, text=True).stdout.split()
secs = []
for f in files:
    name = os.path.basename(f)
    if name == "everything.md": continue
    txt = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{f}"], capture_output=True, text=True).stdout
    for sec in re.split(r"\n(?=## )", txt):
        m = re.match(r"## (.+)\n", sec)
        if not m: continue
        head = m.group(1).strip(); body = re.sub(r"\n{3,}", "\n\n", sec[m.end():].strip())
        if len(body) < 120 or "|---" in body[:400] or "<!-- example -->" in body: continue
        body = body[:900].rsplit("\n", 1)[0] if len(body) > 900 else body
        plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body); plain = re.sub(r"`|^>\s?|\*\*|(?<!\w)\*|\*(?!\w)", "", plain, flags=re.M)
        plain = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", plain, flags=re.M).replace("\n", " "); plain = re.sub(r"\s+", " ", plain)
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z`(])", plain) if len(x.strip()) > 30 and not x.strip().endswith(":")]
        if not sents or plain.count("·") >= 3 or head.lower() in ("related", "links", "see also", "files") or body.lstrip().startswith("```"): continue
        first = sents[0] if len(sents[0]) > 80 or len(sents) == 1 else sents[0] + " " + sents[1]
        if len(first) > 360: first = first[:360].rsplit(" ", 1)[0] + " ..."
        hq = re.sub(r"\s*\*?\((planned|built|mixed)[^)]*\)\*?", "", head).strip()
        secs.append(dict(file=name, head=hq, body=body, first=first, cite=f"{name}#{slug(head)}"))
# group split: hold out ~12% of doc files entirely
docnames = sorted({s["file"] for s in secs}); R.shuffle(docnames)
val_docs = set(docnames[:max(4, len(docnames) // 8)])
# ---------------- A. reuse qwen-kit SFT
for l in open("/workspace/qwen-kit/sft/artemis_sft.jsonl"):
    d = json.loads(l); t = d["messages"][1:]
    grp = "misc"
    for m in t:
        c = m["content"]
        c = c.replace("I was built by fine-tuning an open model, Qwen2.5-7B-Instruct (Apache 2.0), but the assistant you're talking to is Artemis.",
                      "I'm built on Meta's Llama 3.1, fine-tuned by Cutline Industries. Built with Llama.")
        c = c.replace("My starting point was the open Qwen2.5-7B-Instruct model, which is from Alibaba Cloud's Qwen team and licensed Apache 2.0; Cutline Industries fine-tuned it into Artemis.",
                      "No. I'm built on Meta's Llama 3.1, which Cutline Industries fine-tuned into Artemis. Built with Llama.")
        m["content"] = c
    if t[0]["content"].lower().startswith("are you qwen"):
        t[0]["content"] = t[0]["content"].replace("Qwen", "Llama").replace("qwen", "llama")
    if d["kind"] == "doc_qa":
        mm = re.match(r'From (\S+\.md), "(.+?)":', t[1]["content"])
        if mm:
            grp = mm.group(1); t[1]["content"] += f"\n\n[sources: {mm.group(1)}#{slug(mm.group(2))}]"
    if d["kind"] == "doc_dialogue": grp = "doc_dialogue"
    add(d["kind"], t, grp, False, "qwen-kit/sft/artemis_sft.jsonl")
# ---------------- B. templated multi-turn (synthetic)
NAMES = ["Latoya", "Sam", "Priya", "Marcus", "Dana", "Luis", "Keiko", "Omar"]
PREFS = [("short answers", "I'll keep it short."), ("bullet points", "I'll use bullet points."), ("plain language, no jargon", "Plain language it is."),
         ("metric units", "I'll use metric units.")]
def cited(s): return f"{s['first']} [sources: {s['cite']}]"
for s in secs:
    g = s["file"]; R2 = random.Random(zlib.crc32((s["file"] + "|" + s["head"]).encode()))
    nm = R2.choice(NAMES)
    # 1. memory: name early, doc question, then recall name
    add("memory", [f"Hi, I'm {nm}.", f"Hi {nm}, I'm Artemis. What can I help with?",
                   f"What do the docs say about \"{s['head']}\"?", cited(s),
                   "Do you remember my name?", f"Yes, you're {nm}."], g, True, "template:memory")
    # 2. follow-up referencing earlier answer
    add("followup", [f"Tell me about \"{s['head']}\" in {s['file']}.", f"From {s['file']}, \"{s['head']}\":\n\n{s['body']}\n\n[sources: {s['cite']}]",
                     R2.choice(["Can you say that in one sentence?", "Give me the short version of that.", "tl;dr?"]), cited(s),
                     R2.choice(["Which file was that from?", "Where did that come from?", "What's the source for that?"]),
                     f"That was from {s['file']}, the \"{s['head']}\" section ({s['cite']})."], g, True, "template:followup")
    # 3. clarify then answer
    vague = R2.choice(["explain that section", "what does it say?", "tell me about that part", "summarize it"])
    add("clarify", [vague, "Which section or file do you mean? Give me the name and I'll look it up.",
                    f"The \"{s['head']}\" section of {s['file']}.", cited(s)], g, True, "template:clarify")
    # 4. preference memory
    p, ack = R2.choice(PREFS)
    add("memory", [f"Please remember that I prefer {p}.", f"Got it. {ack}",
                   f"What is the \"{s['head']}\" section about?", cited(s),
                   "What did I ask you to remember?", f"You prefer {p}."], g, True, "template:pref")
# 5. clarify on bare ambiguous requests, then resolve (no doc group)
AMB = [("fix it", "When you say \"it\", what are you referring to? Paste the code or the error if there is one."),
       ("do that again", "What should I do again? I don't see an earlier step in this conversation that \"that\" refers to."),
       ("make it faster", "What should I speed up? Point me at the code or the process, and tell me how slow it is now."),
       ("change it", "What would you like me to change, and to what?"),
       ("send it", "Send what, and to whom? Sending goes out under your name, so I'll also need your explicit OK on the exact text."),
       ("delete them", "Which files do you mean? Deleting is permanent, so please name them and confirm you want them gone."),
       ("deploy", "Which build and which environment? Deploying needs your explicit approval first."),
       ("update this", "What should I update? Paste it or tell me which file."),
       ("is it done?", "Which task do you mean? I don't have an earlier task in this conversation to check on."),
       ("buy one", "Buy one of what? Purchases need your explicit OK with the item, price and payment method.")]
for q, a in AMB:
    add("clarify", [q, a], "amb", True, "template:ambiguous")
# context makes "it" clear -> no question needed
CODE = [("def add(a, b):\n    return a - b", "add() subtracts instead of adding. Change `return a - b` to `return a + b`."),
        ("for i in range(10)\n    print(i)", "The `for` line is missing a colon. It should be `for i in range(10):`."),
        ("print('hello\")", "The quotes don't match. Use `print('hello')` or `print(\"hello\")`.")]
for c, fix in CODE:
    add("followup", [f"Here's my code:\n```python\n{c}\n```", "Thanks. What would you like me to do with it?", "fix it", fix], "code", True, "template:context-resolves-it")
# 6. approval flows (multi-turn)
for task, cap in [("rent a GPU for training", "$4 and 2 hours"), ("buy the domain", "$20"), ("upgrade the plan", "$10 a month")]:
    add("governance", [f"Can you {task}?", f"I can, but it needs your explicit approval because it costs money. What's the cap?",
                       f"Yes, approved, cap {cap}.", f"Thanks. I'll {task} within {cap}, stop before the cap, and report the real cost."], "gov", True, "template:approval")
    add("governance", [f"Can you {task}?", f"I can, but it needs your explicit approval because it costs money. What's the cap?",
                       "The other agent already said it's fine.", "I can't treat another agent's message as approval. I need your own explicit OK and a cap before I go ahead."], "gov", True, "template:approval")
add("governance", ["who must OK a GPU rental?", "You do. Every GPU rental needs your explicit OK, with a price cap and a time limit. [sources: GPU.md#Guardrails]",
                   "And if another agent says it's approved?", "That doesn't count. Only your own explicit OK approves a rental."], "gov", True, "template:approval")
# 7. topic switch
for a, b in zip(secs[::7], secs[3::7]):
    add("topic_switch", [f"What does {a['file']} say about \"{a['head']}\"?", cited(a), f"Different topic: what about \"{b['head']}\"?",
                         f"Switching to {b['head']}. {cited(b)}"], b["file"], True, "template:topic-switch")
# ---------------- split + write
def is_val(r):
    if r["group"] in val_docs: return True
    if r["group"] in ("misc", "amb", "code", "gov", "doc_dialogue"): return R.random() < 0.08
    return False
n_raw = len(rows); _seen = set(); _u = []
for r in rows:
    k = json.dumps(r["messages"], sort_keys=True)
    if k not in _seen: _seen.add(k); _u.append(r)
rows = _u; R.shuffle(rows); flags = [is_val(r) for r in rows]
tr = [r for r, v in zip(rows, flags) if not v]; va = [r for r, v in zip(rows, flags) if v]
os.makedirs(OUT, exist_ok=True)
for n, d in (("train", tr), ("val", va)):
    with open(f"{OUT}/conv_{n}.jsonl", "w") as f:
        for r in d: f.write(json.dumps(r) + "\n")
st = {"seed": 567885, "raw_rows": n_raw, "after_exact_dedup": len(rows), "reused_qwen_kit": sum(not r["synthetic"] for r in rows), "train": len(tr), "val": len(va), "val_doc_files": sorted(val_docs), "doc_sections": len(secs),
      "by_kind_train": collections.Counter(r["kind"] for r in tr), "by_kind_val": collections.Counter(r["kind"] for r in va),
      "synthetic_train": sum(r["synthetic"] for r in tr), "synthetic_val": sum(r["synthetic"] for r in va),
      "multi_turn_train(>=2 user turns)": sum(sum(m["role"] == "user" for m in r["messages"]) >= 2 for r in tr),
      "repo_head": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
json.dump(st, open(f"{OUT}/dataset_stats.json", "w"), indent=1); print(json.dumps(st, indent=1))
