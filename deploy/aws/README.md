# AWS GPU box

Self-hosted 3D Body Pose NIM on a `g6.xlarge` (L4, 24 GB) in us-west-2,
instance `i-09ab959e8eba818af`. Temporary home: the work moves to Nebius at the
end, so nothing in `src/snow/` knows about AWS.

## Use it

```bash
aws ec2 start-instances --region us-west-2 --instance-ids i-09ab959e8eba818af
aws ec2 wait instance-running --region us-west-2 --instance-ids i-09ab959e8eba818af
IP=$(aws ec2 describe-instances --region us-west-2 --instance-ids i-09ab959e8eba818af \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)   # changes every start
ssh -i ~/.ssh/snow-gpu.pem -f -N -L 18001:127.0.0.1:8001 ubuntu@$IP
# NIM takes ~1 min after boot to report ready (model is cached in /opt/nim-cache)
SNOW_NIM_PREVIEW=0 SNOW_NIM_TARGET=127.0.0.1:18001 snow pose run1
```

SSH is open to one home IP only (security group `snow-gpu`); if the IP changes,
update the rule. The NIM binds to localhost on the box, so the tunnel is the
only way in.

## Spend guards (installed by `user-data.sh`)

- **spend-meter**: every 5 min adds this instance's on-demand rate to
  `s3://snow-artifacts-009160029345/spend/<YYYY-MM>.txt` and powers off at
  $85/month of compute (the $100 budget minus EBS and S3). Shared by every
  snow GPU instance, so a later `g6e` render box counts against the same cap.
- **idle-stop**: powers off after 30 min with no SSH connection and the GPU
  under 5%.

Power-off means *stopped*, not terminated: the 100 GB volume (~$8/month) stays.

## Pieces

| Resource | Name |
|---|---|
| Role / instance profile | `/app/snow-gpu` (with `app-workload-boundary`), policy in `role-policy.json` |
| NVIDIA key | SSM SecureString `/app/snow/NVIDIA_API_KEY` |
| Bucket | `snow-artifacts-009160029345` (private) |
| Key pair | `snow-gpu`, private key at `~/.ssh/snow-gpu.pem` |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04), driver 595 |
