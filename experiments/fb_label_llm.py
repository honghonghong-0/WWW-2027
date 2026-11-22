"""Seeker-feedback labels (fixed before looking at any label, 2026-11-17).
Unit = (request in VALID or TEST, item i in its community gold, the seeker's own later replies in that thread that mention i).
Only gold items are used (fb item linking is noisy: character names get linked to movies; restricting to recommended items removes most).
Labeler: Llama-3.1-8B-Instruct (vllm, greedy; free text parsed by label prefix, unparsed -> OTHER and counted). Input: request (first 700 chars), movie title, seeker replies (raw, <=1200 chars).
Labels:
  WATCHED_LIKED     seeker watched it (now or before) and liked it / it fit           -> L4 positive
  WATCHED_DISLIKED  seeker watched it and disliked it / it did not fit                -> L4 negative
  WILL_WATCH        not watched yet; interested, will watch, added to list           -> L3 positive (perceived affordance)
  NOT_INTERESTED    not watched; rejects it                                           -> L3 negative
  OTHER             anything else (thanks only, unclear, about another movie)
Second field EXPERIENCE_DESCRIBED yes/no: does the reply say how the movie felt to watch (tense, sad, funny, creepy...)."""
import os, json, collections
os.environ.setdefault("HF_HOME", "/path/to/base/cache/hf")
from vllm import LLM, SamplingParams
W = "/path/to/base/www27"; P = f"{W}/data/reddit_movie/prep_large"
TH = json.load(open(f"{P}/threads.json")); NAME = json.load(open(f"{W}/data/reddit_movie/large/id2name.json"))
units = []
for sp in ("valid", "test"):
    for l in open(f"{P}/requests_{sp}.jsonl"):
        x = json.loads(l); fb = TH[x["req"]]["fb"]
        for i in sorted(set(x["gold"])):
            ts = [f["raw"] for f in fb if i in f["items"]]
            if ts: units.append(dict(split=sp, req=x["req"], user=x["user"], item=i, title=NAME.get(i, i), req_raw=x["raw"][:700], fb_raw=" || ".join(ts)[:1200]))
print("units", collections.Counter(u["split"] for u in units), flush=True)
LAB = ["WATCHED_LIKED", "WATCHED_DISLIKED", "WILL_WATCH", "NOT_INTERESTED", "OTHER"]
SYS = ("You label Reddit movie-request threads. The SEEKER asked for movie recommendations; others recommended movies. "
       "Read the seeker's own later replies and decide the seeker's reaction to ONE recommended movie.")
def q1(u): return (f"SEEKER REQUEST:\n{u['req_raw']}\n\nRECOMMENDED MOVIE: {u['title']}\n\nSEEKER'S LATER REPLIES (mentioning this movie):\n{u['fb_raw']}\n\n"
    "Which label fits the seeker's reaction to THIS movie?\n"
    "WATCHED_LIKED = has watched it (now or earlier) and liked it or says it fit.\nWATCHED_DISLIKED = has watched it and disliked it or says it did not fit.\n"
    "WILL_WATCH = has not watched it yet and is interested / will watch / added it.\nNOT_INTERESTED = has not watched it and rejects it.\n"
    "OTHER = only thanks, unclear, or about something else.\nAnswer with the label only.")
def q2(u): return (f"SEEKER'S REPLY:\n{u['fb_raw']}\n\nDoes the reply describe how the movie {u['title']} FELT to watch (an emotional or experiential quality, "
    "e.g. tense, sad, funny, creepy, heartwarming, boring, intense)? Answer yes or no.")
llm = LLM("NousResearch/Meta-Llama-3.1-8B-Instruct", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.45, seed=0)
tok = llm.get_tokenizer()
chat = lambda s: tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)
o1 = llm.generate([chat(q1(u)) for u in units], SamplingParams(temperature=0, max_tokens=8))
o2 = llm.generate([chat(q2(u)) for u in units], SamplingParams(temperature=0, max_tokens=3))
def parse(t, opts):
    t = t.strip().upper().replace(" ", "_")
    for o in sorted(opts, key=len, reverse=True):
        if t.startswith(o.upper()): return o
    return None
bad = 0
for u, a, b in zip(units, o1, o2):
    l = parse(a.outputs[0].text, LAB); bad += l is None; u["label"] = l or "OTHER"; u["raw_out"] = a.outputs[0].text
    u["exp_described"] = parse(b.outputs[0].text, ["yes", "no"]) or "no"
print("UNPARSED_LABELS", bad, "of", len(units))
json.dump(units, open(f"{W}/results/fb_labels.json", "w"))
print("labels", collections.Counter((u["split"], u["label"]) for u in units)); print("exp_described", collections.Counter(u["exp_described"] for u in units)); print("FB_DONE")
