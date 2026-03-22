#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# Colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        NSE Simulator Launcher        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""

# ── Check prerequisites ───────────────────────────────────────────────────────

if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
  echo -e "${RED}Error: Python not found. Install Python 3.11+${NC}"
  exit 1
fi

if ! command -v node &>/dev/null; then
  echo -e "${RED}Error: Node.js not found. Install Node.js 18+${NC}"
  exit 1
fi

# ── Backend setup ─────────────────────────────────────────────────────────────

echo -e "${YELLOW}Setting up backend...${NC}"
cd "$BACKEND_DIR"

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo -e "${YELLOW}Created backend/.env from .env.example. Edit it to add your API keys.${NC}"
  fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d .venv ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv || python -m venv .venv
fi

# Activate venv
source .venv/bin/activate 2>/dev/null || . .venv/bin/activate

# Install/upgrade dependencies
echo "Installing Python dependencies (this may take a moment on first run)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo -e "${GREEN}Backend ready.${NC}"

# ── Frontend setup ─────────────────────────────────────────────────────────────

echo -e "${YELLOW}Setting up frontend...${NC}"
cd "$FRONTEND_DIR"

if [ ! -d node_modules ]; then
  echo "Installing Node.js dependencies..."
  npm install --legacy-peer-deps
fi

echo -e "${GREEN}Frontend ready.${NC}"

# ── Launch both services ──────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}Starting services...${NC}"
echo -e "  Backend:  ${YELLOW}http://localhost:8000${NC}"
echo -e "  Frontend: ${YELLOW}http://localhost:3000${NC}"
echo -e "  API Docs: ${YELLOW}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}Note: First stock addition will download the FinBERT model (~500MB).${NC}"
echo "Press Ctrl+C to stop both services."
echo ""

# Cleanup on exit
cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend
cd "$BACKEND_DIR"
source .venv/bin/activate 2>/dev/null || . .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

# Wait for both
wait "$BACKEND_PID" "$FRONTEND_PID"
