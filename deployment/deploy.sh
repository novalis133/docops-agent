#!/bin/bash
# DocOps Agent Deployment Script

set -e

echo "DocOps Agent Deployment"
echo "========================"

# Check for required tools
command -v docker >/dev/null 2>&1 || { echo "Error: docker is required"; exit 1; }

# Parse arguments
DEPLOY_TARGET=${1:-local}

case $DEPLOY_TARGET in
    local)
        echo "Deploying locally with Docker Compose..."
        cd "$(dirname "$0")/.."
        docker-compose up -d
        echo ""
        echo "Services started:"
        echo "  - Elasticsearch: http://localhost:9200"
        echo "  - Kibana: http://localhost:5601"
        echo ""
        echo "Run 'python scripts/setup_elasticsearch.py' to create indices"
        ;;

    build)
        echo "Building Docker image..."
        cd "$(dirname "$0")/.."
        docker build -f deployment/Dockerfile -t docops-agent:latest .
        echo "Image built: docops-agent:latest"
        ;;

    digitalocean)
        echo "Deploying to DigitalOcean App Platform..."
        command -v doctl >/dev/null 2>&1 || { echo "Error: doctl is required for DO deployment"; exit 1; }

        cd "$(dirname "$0")"
        doctl apps create --spec do-app-spec.yaml
        echo "App deployment initiated. Check status with: doctl apps list"
        ;;

    *)
        echo "Usage: $0 [local|build|digitalocean]"
        echo ""
        echo "  local        - Start local services with Docker Compose"
        echo "  build        - Build Docker image"
        echo "  digitalocean - Deploy to DigitalOcean App Platform"
        exit 1
        ;;
esac

echo ""
echo "Deployment complete!"
