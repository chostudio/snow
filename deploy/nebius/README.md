# Nebius Cloud GPU Backend & Model Deployment

Nebius Compute deployment for the 3D Body Pose extraction NIM, Model A (Segmenter), Model B (Factored Trick Classifier), and Model C (FIS Scoring).

## Architecture

Nebius provides cost-effective GPU instances (NVIDIA L4 / H100 / L40S) with S3-compatible Object Storage.

- **NIM Body Pose**: Containerized NVIDIA 3D Body Pose NIM running on Nebius GPU compute.
- **Trained Models**:
  - Model A (`segmenter.pt`): Frame-level airborne temporal CNN.
  - Model B (`classifier.pt`): 5-head factored trick classifier.
  - Model C (`model.joblib`): Big Air score regressor & landing classifier.
- **Storage**: Artifacts, dataset bundles, and models synced to Nebius Object Storage (`s3cmd` / `aws s3` endpoint `storage.nebius.cloud`).

## Nebius GPU Setup (`startup.sh`)

Deploying the backend on a Nebius VM:

```bash
# 1. Initialize Nebius CLI and authenticate
nebius compute instance create \
  --name snow-nebius-gpu \
  --zone eu-north1-c \
  --platform gpu-l4 \
  --cores 8 \
  --memory 32GB \
  --gpus 1 \
  --network-interface-spec subnet-id=<SUBNET_ID>,enable-nat=true \
  --boot-disk-spec size=100GB,type=network-ssd \
  --user-data-file deploy/nebius/startup.sh

# 2. Forward NIM gRPC and HTTP API
IP=$(nebius compute instance get snow-nebius-gpu --format json | jq -r .network_interfaces[0].primary_v4_address.address)
ssh -i ~/.ssh/id_rsa -N -L 18001:127.0.0.1:8001 -L 18000:127.0.0.1:8000 ubuntu@$IP

# 3. Direct snow pipeline to Nebius endpoint
export SNOW_NIM_TARGET=127.0.0.1:18001
export SNOW_NIM_PREVIEW=0
snow features run1 --segmenter data/weights/segmenter.pt
```

## Dual Hosting & Compatibility

- **AWS Hosted Deployment**: Preserved intact on the `aws-deployment` branch / `aws-hosted-release` tag for backward compatibility and live AWS endpoints.
- **Nebius Deployment**: Primary target on `main` for cost-effective GPU scaling and sequence model training.
