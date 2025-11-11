#!/bin/bash
# Restore PostgreSQL database from backup file
# Usage: ./scripts/restore_db.sh <backup_file>
#
# WARNING: This will replace all existing data in the database!

set -e

if [ -z "$1" ]; then
  echo "Error: Backup file required"
  echo "Usage: ./scripts/restore_db.sh <backup_file>"
  exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
  echo "Error: Backup file not found: $BACKUP_FILE"
  exit 1
fi

echo "WARNING: This will replace all existing data in the database!"
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
  echo "Restore cancelled."
  exit 0
fi

echo "Restoring database from: $BACKUP_FILE"

# Check if backup is compressed
if [[ "$BACKUP_FILE" == *.gz ]]; then
  echo "Decompressing backup..."
  TEMP_FILE=$(mktemp)
  gunzip -c "$BACKUP_FILE" > "$TEMP_FILE"
  BACKUP_FILE="$TEMP_FILE"
  trap "rm -f $TEMP_FILE" EXIT
fi

# Ensure PostgreSQL is running
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
timeout=30
while ! docker-compose exec -T postgres pg_isready -U digitalme > /dev/null 2>&1; do
  sleep 1
  timeout=$((timeout - 1))
  if [ $timeout -eq 0 ]; then
    echo "Error: PostgreSQL did not become ready in time"
    exit 1
  fi
done

# Restore the backup
echo "Restoring database..."
# Use postgres database to allow CREATE DATABASE commands
docker-compose exec -T postgres psql \
  -U digitalme \
  -d postgres \
  < "$BACKUP_FILE" || {
  echo "Note: If backup includes CREATE DATABASE, restoring to 'digitalme' database directly..."
  docker-compose exec -T postgres psql \
    -U digitalme \
    -d digitalme \
    < "$BACKUP_FILE"
}

echo "Database restore completed successfully!"

