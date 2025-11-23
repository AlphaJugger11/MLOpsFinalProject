#!/bin/bash
# scripts/push_images.sh
# Script to push Docker images to DockerHub or AWS ECR

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Pushing FL Docker Images ===${NC}"

# Determine registry type (dockerhub or ecr)
REGISTRY_TYPE=${REGISTRY_TYPE:-"dockerhub"}
echo -e "${YELLOW}Registry type: ${REGISTRY_TYPE}${NC}"

# Get version
if command -v git &> /dev/null && [ -d .git ]; then
    VERSION=$(git rev-parse --short HEAD)
else
    VERSION="latest"
fi

# Function to push to DockerHub
push_to_dockerhub() {
    echo -e "${GREEN}Authenticating with DockerHub...${NC}"
    
    # Check if credentials are provided
    if [ -z "$DOCKERHUB_USERNAME" ] || [ -z "$DOCKERHUB_TOKEN" ]; then
        echo -e "${RED}Error: DOCKERHUB_USERNAME and DOCKERHUB_TOKEN must be set${NC}"
        echo -e "${YELLOW}Usage: DOCKERHUB_USERNAME=user DOCKERHUB_TOKEN=token ./scripts/push_images.sh${NC}"
        exit 1
    fi
    
    # Login to DockerHub
    echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ DockerHub authentication failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ DockerHub authentication successful${NC}"
    
    # Tag images with username prefix
    REGISTRY_PREFIX="${DOCKERHUB_USERNAME}/"
    
    docker tag fl-server:latest ${REGISTRY_PREFIX}fl-server:${VERSION}
    docker tag fl-server:latest ${REGISTRY_PREFIX}fl-server:latest
    docker tag fl-client:latest ${REGISTRY_PREFIX}fl-client:${VERSION}
    docker tag fl-client:latest ${REGISTRY_PREFIX}fl-client:latest
    
    # Push server images
    echo -e "${GREEN}Pushing server images...${NC}"
    docker push ${REGISTRY_PREFIX}fl-server:${VERSION}
    docker push ${REGISTRY_PREFIX}fl-server:latest
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Server images pushed successfully${NC}"
    else
        echo -e "${RED}✗ Server image push failed${NC}"
        exit 1
    fi
    
    # Push client images
    echo -e "${GREEN}Pushing client images...${NC}"
    docker push ${REGISTRY_PREFIX}fl-client:${VERSION}
    docker push ${REGISTRY_PREFIX}fl-client:latest
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Client images pushed successfully${NC}"
    else
        echo -e "${RED}✗ Client image push failed${NC}"
        exit 1
    fi
    
    # Verify images exist in registry
    echo -e "${GREEN}Verifying images in DockerHub...${NC}"
    docker pull ${REGISTRY_PREFIX}fl-server:latest > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Server image verified in registry${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Could not verify server image${NC}"
    fi
    
    docker pull ${REGISTRY_PREFIX}fl-client:latest > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Client image verified in registry${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Could not verify client image${NC}"
    fi
}

# Function to push to AWS ECR
push_to_ecr() {
    echo -e "${GREEN}Authenticating with AWS ECR...${NC}"
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}Error: AWS CLI is not installed${NC}"
        exit 1
    fi
    
    # Get AWS account ID and region
    AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
    AWS_REGION=${AWS_REGION:-"us-east-1"}
    
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        echo -e "${RED}Error: Could not determine AWS account ID${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}AWS Account: ${AWS_ACCOUNT_ID}${NC}"
    echo -e "${YELLOW}AWS Region: ${AWS_REGION}${NC}"
    
    # Login to ECR
    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ ECR authentication failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ ECR authentication successful${NC}"
    
    # Create ECR repositories if they don't exist
    aws ecr describe-repositories --repository-names fl-server --region ${AWS_REGION} > /dev/null 2>&1 || \
        aws ecr create-repository --repository-name fl-server --region ${AWS_REGION}
    
    aws ecr describe-repositories --repository-names fl-client --region ${AWS_REGION} > /dev/null 2>&1 || \
        aws ecr create-repository --repository-name fl-client --region ${AWS_REGION}
    
    # Tag images with ECR prefix
    REGISTRY_PREFIX="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/"
    
    docker tag fl-server:latest ${REGISTRY_PREFIX}fl-server:${VERSION}
    docker tag fl-server:latest ${REGISTRY_PREFIX}fl-server:latest
    docker tag fl-client:latest ${REGISTRY_PREFIX}fl-client:${VERSION}
    docker tag fl-client:latest ${REGISTRY_PREFIX}fl-client:latest
    
    # Push server images
    echo -e "${GREEN}Pushing server images to ECR...${NC}"
    docker push ${REGISTRY_PREFIX}fl-server:${VERSION}
    docker push ${REGISTRY_PREFIX}fl-server:latest
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Server images pushed successfully${NC}"
    else
        echo -e "${RED}✗ Server image push failed${NC}"
        exit 1
    fi
    
    # Push client images
    echo -e "${GREEN}Pushing client images to ECR...${NC}"
    docker push ${REGISTRY_PREFIX}fl-client:${VERSION}
    docker push ${REGISTRY_PREFIX}fl-client:latest
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Client images pushed successfully${NC}"
    else
        echo -e "${RED}✗ Client image push failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Images verified in ECR${NC}"
}

# Main execution
case $REGISTRY_TYPE in
    dockerhub)
        push_to_dockerhub
        ;;
    ecr)
        push_to_ecr
        ;;
    *)
        echo -e "${RED}Error: Invalid REGISTRY_TYPE. Must be 'dockerhub' or 'ecr'${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}=== Push Complete ===${NC}"
echo -e "Images pushed with tags:"
echo -e "  - ${VERSION}"
echo -e "  - latest"
