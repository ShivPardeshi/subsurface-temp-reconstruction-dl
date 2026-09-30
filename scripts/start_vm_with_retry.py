"""Retry starting oceanembed-l4-training VM until capacity opens up in us-central1-a."""

import subprocess
import time
import sys

def start_vm(max_attempts=6, wait_seconds=15):
    print("Attempting to start GCP VM oceanembed-l4-training in us-central1-a...")
    for attempt in range(1, max_attempts + 1):
        print(f"\n[Attempt {attempt}/{max_attempts}] Running: gcloud compute instances start oceanembed-l4-training --zone=us-central1-a")
        res = subprocess.run(
            "gcloud compute instances start oceanembed-l4-training --zone=us-central1-a",
            shell=True,
            capture_output=True,
            text=True
        )
        if res.returncode == 0:
            print("\n[SUCCESS] Instance oceanembed-l4-training successfully STARTED!")
            print(res.stdout)
            return True
        else:
            err_snippet = res.stderr.strip()
            if "ZONE_RESOURCE_POOL_EXHAUSTED" in err_snippet:
                print(f"  Resource pool temporarily exhausted in us-central1-a. Waiting {wait_seconds}s before retry...")
            else:
                print(f"  Error: {err_snippet[:200]}")
            if attempt < max_attempts:
                time.sleep(wait_seconds)

    print("\n[Timeout] Could not acquire L4 instance in us-central1-a after retries.")
    return False

if __name__ == "__main__":
    success = start_vm()
    sys.exit(0 if success else 1)
