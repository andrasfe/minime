#!/bin/bash
# Script to run Docker integration tests

set -e

echo "Building and starting Docker containers..."
docker-compose -f docker-compose.yml up -d --build

echo "Waiting for services to be ready..."
sleep 10

echo "Running Docker integration tests..."
pytest tests/test_docker_integration.py -m docker -v

echo "Tests completed. Containers are still running."
echo "To stop containers: docker-compose -f docker-compose.yml down"
echo "To stop and remove volumes: docker-compose -f docker-compose.yml down -v"

