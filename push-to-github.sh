#!/bin/bash
set -e

cd /Users/priyuu/Desktop/ai/logistics-delay-prediction-main

# Configure git
git config user.email "imnextin@github.com"
git config user.name "imnextin"

# Check if already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git remote add origin https://github.com/imnextin/amazon-logistics-ai-predictor.git
    git add -A
    git commit -m "Initial commit: Add logistics delay prediction project files"
fi

# Set the merge strategy and push
export GIT_EDITOR=nano
git config pull.rebase false

# Try to pull first
git pull origin main --allow-unrelated-histories || true

# Push to repository
git push -u origin main

echo "Successfully pushed to GitHub!"
