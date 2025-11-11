#!/bin/bash
# Backup PostgreSQL database from Docker container
# Usage: ./scripts/backup_db.sh [backup_file]

set -e

BACKUP_DIR="${BACKUP_DIR:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${1:-${BACKUP_DIR}/backup_${TIMESTAMP}.sql}"

# Create backup directory if it doesn't exist
mkdir -p "$(dirname "$BACKUP_FILE")"

echo "Creating database backup..."
echo "Backup file: $BACKUP_FILE"

# Use pg_dump for SQL backup (recommended)
docker-compose exec -T postgres pg_dump \
  -U digitalme \
  -d digitalme \
  --clean \
  --if-exists \
  --create \
  > "$BACKUP_FILE"

# Compress the backup
if command -v gzip &> /dev/null; then
  echo "Compressing backup..."
  gzip -f "$BACKUP_FILE"
  BACKUP_FILE="${BACKUP_FILE}.gz"
  echo "Compressed backup: $BACKUP_FILE"
fi

echo "Backup completed successfully!"
echo "Backup size: $(du -h "$BACKUP_FILE" | cut -f1)"
echo ""
echo "To restore, use:"
echo "  ./scripts/restore_db.sh $BACKUP_FILE"

