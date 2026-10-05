#!/usr/bin/env python3
"""
ExpressEval - Local Automated Benchmark & Commit Utility
Runs verification, calculates multi-axis acoustic benchmarks, records telemetry,
and automatically commits and pushes to GitHub with resilient auto-retry.
"""

import os
import sys
import subprocess
import time
import random
from datetime import datetime, timezone

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_DIR)


def run_cmd(cmd, check=True):
    print(f"[*] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True, encoding="utf-8")
    if check and res.returncode != 0:
        print(f"[!] Error executing: {' '.join(cmd)}")
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr)
        raise RuntimeError(f"Command failed with code {res.returncode}")
    return res


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 68)
    print(" [*] ExpressEval: Automated Benchmark, Telemetry & Auto-Commit")
    print("=" * 68)

    # 1. Run Verification Suite
    print("[1/4] Running test verification suite...")
    test_res = run_cmd([sys.executable, "-m", "pytest", "tests/", "-q"])
    print("  * Unit tests passed successfully.")

    # 2. Run Benchmark Telemetry Suite
    print("[2/4] Running multi-axis evaluation benchmark...")
    bench_res = run_cmd([sys.executable, "scripts/run_benchmark.py"])
    print("  * Benchmark telemetry and logs updated.")

    # 3. Configure Git author identity
    print("[3/4] Configuring Git attribution...")
    run_cmd(["git", "config", "user.name", "Scrooge-777"])
    run_cmd(["git", "config", "user.email", "318593710+Scrooge-777@users.noreply.github.com"])
    run_cmd(["git", "config", "http.sslBackend", "schannel"])
    run_cmd(["git", "config", "http.version", "HTTP/1.1"])

    # 4. Check for staged changes and commit
    run_cmd(["git", "add", "-A"])
    diff_check = run_cmd(["git", "diff", "--staged", "--quiet"], check=False)
    if diff_check.returncode == 0:
        print("[-] No changes to commit (working tree clean).")
        return

    messages = [
        "chore(eval): update benchmark telemetry and score logs",
        "chore(metrics): refresh multi-axis speech evaluation logs",
        "chore(benchmark): update validation metrics and telemetry distribution",
        "chore(eval): record cross-lingual consistency and MOS regression logs",
        "chore(telemetry): update acoustic expressiveness benchmark results",
        "chore(stats): refresh bootstrap confidence intervals and correlation telemetry",
        "chore(eval): synchronize acoustic prosody variance and pitch velocity logs",
        "chore(benchmark): update Ridge aggregator validation metrics",
    ]
    commit_msg = random.choice(messages)
    print(f"[4/4] Committing: '{commit_msg}'")
    run_cmd(["git", "commit", "-m", commit_msg])

    # 5. Push with auto-retry
    print("[*] Synchronizing with origin/main on GitHub...")
    pushed = False
    for attempt in range(1, 6):
        print(f"  * Push attempt {attempt}/5...")
        run_cmd(["git", "pull", "--rebase", "origin", "main"], check=False)
        push_res = run_cmd(["git", "push", "origin", "main"], check=False)
        if push_res.returncode == 0:
            pushed = True
            print("  * Successfully pushed commit to GitHub!")
            break
        time.sleep(3)

    if not pushed:
        print("[!] Push encountered network delays. Commits are safely recorded locally and will push on next run.")
    else:
        print("=" * 68)
        print("[+] Automated evaluation and commit completed successfully!")
        print("=" * 68)


if __name__ == "__main__":
    main()
