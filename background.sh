#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Usage: ./run_in_background.sh <script.py> [script args...]
if [ -z "$1" ] ; then
    echo "Usage: ./background.sh <script.py> [script args...]"
    echo "Example: ./background.sh src/swstl.py -b 10 -s 10 -t 10"
    exit 1
fi
script="$1"
output="$1.log"
shift 1  # Remove first arg, leaving only script arguments
script_args="$@"
# Check if script exists
if [ ! -f "$script" ]; then
    echo "Error: Script '$script' not found"
    exit 1
fi
# Check for existing processes running this script
scriptname=$(basename "$script")
existing=$(ps aux | grep "[p]ython.*$scriptname" | grep -v grep)
if [ -n "$existing" ]; then
    echo "Found running processes for $scriptname:"
    ps aux | grep "[p]ython.*$scriptname" | grep -v grep
    echo ""
    read -p "Kill existing processes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "$scriptname"
        sleep 1
        echo "Killed existing processes."
    else
        echo "Exiting without starting new process."
        exit 1
    fi
fi
# Start new process
echo -e "${CYAN}Starting $script with args: $script_args"
echo "Output: $output"
PYTHONUNBUFFERED=1 nohup uv run "$script" $script_args > "$output" 2>&1 &
pid=$!
echo ""
echo -e "Started with PID: ${GREEN}$pid${CYAN}"
echo ""
echo -e "Monitor with:  ${YELLOW}tail -f $output${CYAN}"
echo -e "Kill with:     ${RED}kill $pid${NC}"
tail -f "$output"
