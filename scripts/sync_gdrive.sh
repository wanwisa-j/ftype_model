#!/usr/bin/env bash
set -euo pipefail

# usage: sync_gdrive.sh --remote NAME --id FOLDER_ID --dest /path/to/dir [--yes]
# If --yes is provided the script will run the real sync after the dry-run.

RCLONE=${RCLONE:-$HOME/bin/rclone}
REMOTE="tor-gdrive"
FOLDER_ID=""
DEST="/fs2/ftype_mflf/shared_data/gdrive"
DO_SYNC=false

print_usage(){
  echo "Usage: $0 --remote NAME --id FOLDER_ID [--dest PATH] [--yes]"
  echo "  --remote NAME   : rclone remote name (default: tor-gdrive)"
  echo "  --id FOLDER_ID  : Google Drive folder ID to sync (required)"
  echo "  --dest PATH     : local destination directory (default: $DEST)"
  echo "  --yes           : run the real sync after dry-run"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2;;
    --id) FOLDER_ID="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --yes) DO_SYNC=true; shift 1;;
    -h|--help) print_usage;;
    *) echo "Unknown arg: $1"; print_usage;;
  esac
done

if [[ -z "$FOLDER_ID" ]]; then
  echo "Error: --id FOLDER_ID is required"
  print_usage
fi

if [[ ! -x "$RCLONE" ]]; then
  echo "rclone not found at $RCLONE"
  echo "You can install rclone to ~/bin (the script assumes ~/bin/rclone)."
  echo "Or set RCLONE environment variable to point to your rclone binary."
  exit 2
fi

mkdir -p "$DEST"

echo "Using rclone: $RCLONE"
echo "Remote: $REMOTE"
echo "Folder ID: $FOLDER_ID"
echo "Destination: $DEST"

echo "Checking if remote '$REMOTE' exists..."
if ! "$RCLONE" listremotes | grep -qxE "^${REMOTE}:$"; then
  echo "Remote '$REMOTE' not found."
  echo "Run '$RCLONE config' to create a Google Drive remote, or provide an existing remote name via --remote."
  echo "If you prefer a non-interactive approach, create a Google service account, share the folder with the service account email, and create an rclone remote that uses the service account file."
  exit 3
fi

echo "Running dry-run sync (preview). This will not change local files."
echo "Command: $RCLONE sync \"${REMOTE}:${FOLDER_ID}\" \"${DEST}\" --dry-run --progress --verbose"

"$RCLONE" sync "${REMOTE}:${FOLDER_ID}" "${DEST}" --dry-run --progress --verbose || true

if [ "$DO_SYNC" = true ]; then
  echo "--yes provided; running real sync now."
  echo "This will copy files from the Drive folder to the destination and may delete files locally to match remote."
  read -r -p "Proceed with real sync? (y/N) " ans || true
  if [[ "$ans" =~ ^[Yy]$ ]]; then
    "$RCLONE" sync "${REMOTE}:${FOLDER_ID}" "${DEST}" --progress --verbose
    echo "Sync complete."
  else
    echo "Aborted by user. No changes made."
  fi
else
  echo "Dry-run finished. To perform the real sync, re-run this script with --yes after you review the output."
fi
