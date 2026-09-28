#!/bin/sh
# Applies pending Alembic migrations on every container start before
# running the given command (idempotent -- a no-op if the schema is
# already current, so this is safe to run on every restart).
set -e

echo "Applying database migrations..."
alembic upgrade head

exec "$@"
