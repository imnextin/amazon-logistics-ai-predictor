#!/usr/bin/env python3
import os
import subprocess
import sys

def run_git_command(cmd_list):
    """Run a git command and return the result"""
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, cwd='/Users/priyuu/Desktop/ai/logistics-delay-prediction-main', timeout=30, env={**os.environ, 'GIT_EDITOR': 'true'})
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

# Change to working directory
os.chdir('/Users/priyuu/Desktop/ai/logistics-delay-prediction-main')

# Step 1: Abort any ongoing merge
print("Step 1: Aborting any ongoing merge...")
merge_files = ['.git/MERGE_HEAD', '.git/MERGE_MODE', '.git/MERGE_MSG', '.git/.MERGE_MSG.swp', '.git/AUTO_MERGE', '.git/ORIG_HEAD']
for f in merge_files:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"  Removed {f}")
        except Exception as e:
            print(f"  Could not remove {f}: {e}")

# Step 2: Configure git
print("\nStep 2: Configuring git...")
run_git_command(['git', 'config', 'user.email', 'imnextin@github.com'])
run_git_command(['git', 'config', 'user.name', 'imnextin'])
run_git_command(['git', 'config', 'pull.rebase', 'false'])
print("  Git configured")

# Step 3: Check git status
print("\nStep 3: Checking git status...")
code, out, err = run_git_command(['git', 'status'])
print(f"  Output: {out[:200]}")

# Step 4: Add and commit changes
print("\nStep 4: Adding and committing changes...")
code, out, err = run_git_command(['git', 'add', '-A'])
code, out, err = run_git_command(['git', 'commit', '-m', 'Update project with latest code'])
if 'nothing to commit' in err or 'nothing to commit' in out:
    print("  Nothing new to commit")
else:
    print(f"  Commit: {out[:200]}")

# Step 5: Pull from remote
print("\nStep 5: Pulling from remote...")
code, out, err = run_git_command(['git', 'pull', 'origin', 'main', '--allow-unrelated-histories'])
if code == 0:
    print("  Pull successful")
else:
    print(f"  Pull status: {code}")
    print(f"  Error: {err[:200]}")

# Step 6: Push to remote
print("\nStep 6: Pushing to GitHub...")
code, out, err = run_git_command(['git', 'push', '-u', 'origin', 'main'])
if code == 0:
    print("✓ Successfully pushed to GitHub!")
    print(f"  Output: {out[:200]}")
else:
    print(f"✗ Push failed with code {code}")
    print(f"  Stderr: {err[:300]}")
    print(f"  Stdout: {out[:300]}")

print("\nDone!")
