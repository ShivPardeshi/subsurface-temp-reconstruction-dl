import json
from pathlib import Path
from datetime import datetime

brain_dir = Path(r"C:\Users\123ta\.gemini\antigravity-ide\brain")
convs = []

for cdir in brain_dir.iterdir():
    if not cdir.is_dir():
        continue
    tpath = cdir / ".system_generated" / "logs" / "transcript.jsonl"
    if not tpath.exists():
        continue
    
    first_prompt = None
    last_prompt = None
    count = 0
    
    with open(tpath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("type") == "USER_INPUT":
                    count += 1
                    content = d.get("content", "")
                    # Strip <USER_REQUEST> tags if present
                    clean_content = content.replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "").strip()
                    if first_prompt is None:
                        first_prompt = clean_content
                    last_prompt = clean_content
            except Exception:
                pass
                
    mtime = datetime.fromtimestamp(tpath.stat().st_mtime)
    convs.append({
        "id": cdir.name,
        "mtime": mtime,
        "count": count,
        "first": first_prompt[:100].replace("\n", " ") if first_prompt else "N/A",
        "last": last_prompt[:100].replace("\n", " ") if last_prompt else "N/A"
    })

convs = sorted(convs, key=lambda x: x["mtime"], reverse=True)
print(f"Total conversations found in IDE brain: {len(convs)}\n")
print("=== TOP 10 MOST RECENT CONVERSATIONS ===\n")
for idx, c in enumerate(convs[:10]):
    print(f"[{idx+1}] ID: {c['id']}")
    print(f"    Date & Time: {c['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Total Prompts: {c['count']}")
    print(f"    First Message: {c['first']}")
    print(f"    Last Message:  {c['last']}")
    print("=" * 85)
