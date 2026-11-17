#!/bin/bash
# Re-run a resumable python job until it prints DONE (the host is short on commit memory; random allocation failures happen).
# usage: retry.sh LOG MAX_ATTEMPTS script.py [args...]
log="$1"; max="$2"; shift 2
cd "$(dirname "$0")"
for i in $(seq 1 "$max"); do
  echo "=== attempt $i $(date +%T)" >> "$log"
  ../../third_party_neuralsvb/venv/Scripts/python.exe -I "$@" >> "$log" 2>&1
  grep -q "^DONE" "$log" && exit 0
  sleep 8
done
echo "GAVE_UP" >> "$log"
