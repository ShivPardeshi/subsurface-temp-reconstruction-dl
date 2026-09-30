import json
from pathlib import Path
from datetime import datetime

brain_dir = Path(r"C:\Users\123ta\.gemini\antigravity-ide\brain")
results = []

for tpath in brain_dir.glob("*/.system_generated/logs/transcript.jsonl"):
    cid = tpath.parents[2].name
    with open(tpath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if "MASTER_PROJECT_DOCUMENTATION" in line:
                d = json.loads(line)
                stype = d.get("type", "")
                source = d.get("source", "")
                tool_calls = d.get("tool_calls", [])
                for call in tool_calls:
                    cname = call.get("name", "")
                    cargs = call.get("args", {})
                    if "MASTER_PROJECT_DOCUMENTATION" in str(cargs):
                        results.append({
                            "conversation_id": cid,
                            "step_index": d.get("step_index", idx),
                            "type": stype,
                            "source": source,
                            "tool_name": cname,
                            "target_file": cargs.get("TargetFile") or cargs.get("AbsolutePath") or cargs.get("CommandLine")
                        })

print(f"Found {len(results)} references across conversations:")
for r in results:
    print(f"Conversation: {r['conversation_id']} | Tool: {r['tool_name']} | File: {r['target_file']}")
