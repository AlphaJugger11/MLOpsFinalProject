#!/bin/bash
# scripts/build_images.sh
# Script to build Docker images for FL server and clients

set -e  # Exit on error

# Colors for output
RED='\033[0:31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building FL Docker Images ===${NC}"

# Get version from git commit SHA or use 'latest'
if command -v git &> /dev/null && [ -d .git ]; then
    VERSION=$(git rev-parse --short HEAD)
    echo -e "${YELLOW}Using git commit SHA as version: ${VERSION}${NC}"
else
    VERSION="latest"
    echo -e "${YELLOW}Git not available, using version: ${VERSION}${NC}"
fi

# Registry prefix (can be overridden with environment variable)
REGISTRY_PREFIX=${DOCKER_REGISTRY:-""}
if [ -n "$REGISTRY_PREFIX" ]; then
    echo -e "${YELLOW}Using registry prefix: ${REGISTRY_PREFIX}${NC}"
fi

# Build server image
echo -e "${GREEN}Building server image...${NC}"
docker build \
    -t ${REGISTRY_PREFIX}fl-server:${VERSION} \
    -t ${REGISTRY_PREFIX}fl-server:latest \
    -f server/Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Server image built successfully${NC}"
else
    echo -e "${RED}✗ Server image build failed${NC}"
    exit 1
fi

# Build client image
echo -e "${GREEN}Building client image...${NC}"
docker build \
    -t ${REGISTRY_PREFIX}fl-client:${VERSION} \
    -t ${REGISTRY_PREFIX}fl-client:latest \
    -f client/Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Client image built successfully${NC}"
else
    echo -e "${RED}✗ Client image build failed${NC}"
    exit 1
fi

# Check image sizes
echo -e "${GREEN}=== Image Sizes ===${NC}"
SERVER_SIZE=$(docker images ${REGISTRY_PREFIX}fl-server:latest --format "{{.Size}}")
CLIENT_SIZE=$(docker images ${REGISTRY_PREFIX}fl-client:latest --format "{{.Size}}")

echo -e "Server image: ${SERVER_SIZE}"
echo -e "Client image: ${CLIENT_SIZE}"

# Warn if images are too large (> 2GB)
SERVER_SIZE_MB=$(docker images ${REGISTRY_PREFIX}fl-server:latest --format "{{.Size}}" | sed 's/MB//' | sed 's/GB/*1024/' | bc 2>/dev/null || echo "0")
if [ -n "$SERVER_SIZE_MB" ] && [ "$SERVER_SIZE_MB" != "0" ]; then
    if (( $(echo "$SERVER_SIZE_MB > 2048" | bc -l) )); then
        echo -e "${YELLOW}⚠ Warning: Server image is larger than 2GB${NC}"
    fi
fi

echo -e "${GREEN}=== Build Complete ===${NC}"
echo -e "Server images:"
echo -e "  - ${REGISTRY_PREFIX}fl-server:${VERSION}"
echo -e "  - ${REGISTRY_PREFIX}fl-server:latest"
echo -e "Client images:"
echo -e "  - ${REGISTRY_PREFIX}fl-client:${VERSION}"
echo -e "  - ${REGISTRY_PREFIX}fl-client:latest"
