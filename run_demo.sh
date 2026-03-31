#!/bin/bash
# Cross-Layer Physics-Based Vehicle IDS — Demo Launcher (Linux/macOS)
set -e

cleanup() {
    echo "Shutting down..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

echo "============================================"
echo "  Cross-Layer Physics-Based Vehicle IDS"
echo "  Starting Demo..."
echo "============================================"

echo "[1/2] Starting Dashboard..."
streamlit run dashboard.py &
sleep 3

echo "[2/2] Starting Interactive Demo..."
python quick_demo.py

echo ""
echo "Demo complete! Check the dashboard at http://localhost:8501"
