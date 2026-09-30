import json
from pathlib import Path

tpath = Path(r"C:\Users\123ta\.gemini\antigravity-ide\brain\2ce519f2-2cae-44fa-aa36-2706cc37acf0\.system_generated\logs\transcript.jsonl")
if tpath.exists():
    with open(tpath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            if d.get("type") == "USER_INPUT":
                content = d.get("content", "")
                print(f"--- Prompt at Step {idx} ---")
                print(content[:500])
                print("...\n")
