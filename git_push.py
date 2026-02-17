#!/usr/bin/env python3
import subprocess
import sys
import os

os.chdir('/Users/priyuu/Desktop/ai/logistics-delay-prediction-main')

commands = [
    ['git', 'config', 'user.email', 'imnextin@github.com'],
    ['git', 'config', 'user.name', 'imnextin'],
    ['git', 'config', 'pull.rebase', 'false'],
    ['git', 'add', '-A'],
    ['git', 'commit', '-m', 'Update project files'],
    ['git', 'pull', 'origin', 'main', '--allow-unrelated-histories'],
    ['git', 'push', '-u', 'origin', 'main'],
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            if 'nothing to commit' not in result.stderr and 'already up to date' not in result.stderr.lower():
                print(f"Error running {cmd}")
        else:
            print(f"Success: {result.stdout[:200]}")
    except Exception as e:
        print(f"Exception: {e}")

print("\nGit push complete!")
