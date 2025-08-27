age:
# ./record_sample.sh <pose> <position> <angle> <sample_id>

POSE=$1
POSITION=$2
ANGLE=$3
SAMPLE_ID=$4

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <pose> <position> <angle> <sample_id>"
  exit 1
fi

POSE_FILENAME="${POSE}-${POSITION}-${ANGLE}-${SAMPLE_ID}.dat"
REF_FILENAME="no_person-${POSE}-${POSITION}-${ANGLE}-${SAMPLE_ID}.dat"

record_csi() {
  FILENAME=$1
  DURATION=$2

  # Try timeout if available
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

echo "Recording NO PERSON reference: $REF_FILENAME (5s)"
record_csi "$REF_FILENAME" 5
sleep 2

echo "⚠️  Please assume pose: $POSE at $POSITION facing $ANGLE°"
read -p "Press Enter to start recording..."

echo "Recording POSE sample: $POSE_FILENAME (5s)"
record_csi "$POSE_FILENAME" 5

echo "✅ Saved:"
echo "  → Pose: $POSE_FILENAME"
echo "  → No-person: $REF_FILENAME"

