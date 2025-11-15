#!/bin/bash
# Clear all data from the Digital Me database
# Usage: ./scripts/clear_db.sh
#
# WARNING: This will delete ALL chat summaries and reset the digital_me profile!

set -e

echo "WARNING: This will delete ALL data from the database!"
echo "  - All chat summaries will be deleted"
echo "  - Digital me profile will be reset to empty"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
  echo "Clear cancelled."
  exit 0
fi

echo ""
echo "Clearing database..."

# Clear chat_summaries table and reset digital_me
docker-compose exec -T postgres psql -U digitalme -d digitalme <<EOF
TRUNCATE TABLE chat_summaries;
UPDATE digital_me SET summary_text = '', embedding = NULL, updated_at = CURRENT_TIMESTAMP WHERE id = 1;
EOF

# Verify the clear
echo ""
echo "Verifying database state..."
docker-compose exec -T postgres psql -U digitalme -d digitalme -c "
SELECT 
  (SELECT COUNT(*) FROM chat_summaries) as chat_summaries_count,
  (SELECT COUNT(*) FROM digital_me) as digital_me_count,
  (SELECT LENGTH(summary_text) FROM digital_me WHERE id = 1) as digital_me_length;
"

echo ""
echo "✅ Database cleared successfully!"
echo ""
echo "The database is now empty and ready for new data."

