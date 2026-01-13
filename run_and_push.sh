#!/bin/bash

# Run fetch_nba_data.py with window=5 and window=7, then commit and push

cd "$(dirname "$0")"

echo "=== Running with WINDOW=5 ==="
sed -i '' 's/^WINDOW = .*/WINDOW = 5/' fetch_nba_data.py
python fetch_nba_data.py

echo ""
echo "=== Running with WINDOW=7 ==="
sed -i '' 's/^WINDOW = .*/WINDOW = 7/' fetch_nba_data.py
python fetch_nba_data.py

echo ""
echo "=== Committing and pushing ==="
git add .
git commit -m 'update'
git push

echo ""
echo "=== Done ==="
