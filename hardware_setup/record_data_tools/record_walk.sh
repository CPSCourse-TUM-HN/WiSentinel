age:
# ./record_sample.sh <pose> <position> <angle> <sample_id> <walkout_direction>

POSE=$1
POSITION=$2
ANGLE=$3
SAMPLE_ID=$4
WALKOUT_DIR=$5  # 'left' or 'right'

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <pose> <position> <angle> <sample_id> <walkout_direction>"
  echo "Example: $0 standing pos2 90 1 left"
  exit 1
fi

# Filenames
REF_FILENAME="no_person-walk-${POSE}-${POSITION}-${ANGLE}-${SAMPLE_ID}.dat"
WALKIN_FILENAME="walkin-${POSE}-${POSITION}-${ANGLE}-${SAMPLE_ID}.dat"
WALKOUT_FILENAME="walkout-${POSE}-${POSITION}-${ANGLE}-${SAMPLE_ID}-${WALKOUT_DIR}.dat"

record_csi() {
  FILENAME=$1
  DURATION=$2

  if command -v timeout >/dev/null 2>&1; then
    echo "[INFO] Using timeout to record: $FILENAME"
    timeout "$DURATION" ./recvCSI "$FILENAME"
  else
    echo "[INFO] Using manual timeout fallback for: $FILENAME"
    ./recvCSI "$FILENAME" &
    CSIPID=$!
    sleep "$DURATION"
    kill -SIGTERM "$CSIPID"
    wait "$CSIPID" 2>/dev/null
  fi
}

# 1. No-person calibration
echo "Recording NO PERSON reference: $REF_FILENAME (5s)"
record_csi "$REF_FILENAME" 5
sleep 2

# 2. Walk-in phase
echo "🚶 Please walk in from the LEFT toward position $POSITION, angle $ANGLE°, pose: $POSE"
read -p "Press Enter to start recording walk-in..."
record_csi "$WALKIN_FILENAME" 5
sleep 2

# 3. Walk-out phase
echo "🚶 Now walk OUT to the $WALKOUT_DIR"
read -p "Press Enter to start recording walk-out..."
record_csi "$WALKOUT_FILENAME" 5

# ✅ Summary
echo "✅ Recording complete. Saved files:"
echo "  → No-person:   $REF_FILENAME"
echo "  → Walk-in:     $WALKIN_FILENAME"
echo "  → Walk-out:    $WALKOUT_FILENAME"

