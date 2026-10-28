#!/bin/bash
# Boot script for the snow GPU box. Installs two guards, then starts the NIM.
#   spend-meter: adds this instance's cost to a shared monthly ledger in S3 every
#     5 min and powers off once the month's total reaches the cap.
#   idle-stop: powers off after 30 min with no SSH connection and an idle GPU.
# Shutdown behaviour is "stop", so a poweroff stops billing for compute (the EBS
# volume still costs ~$0.08/GB-month).
set -euo pipefail

cat > /usr/local/bin/snow-spend-meter <<'METER'
#!/bin/bash
# Monthly compute ledger shared by every snow GPU instance; cap is in dollars.
set -uo pipefail
CAP=85            # $100 budget minus ~$15 of EBS and S3
BUCKET=snow-artifacts-009160029345
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
TYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type)
case "$TYPE" in   # us-west-2 on-demand $/hr
  g6.xlarge)   RATE=0.8048 ;;
  g5.xlarge)   RATE=1.006 ;;
  g6e.xlarge)  RATE=1.861 ;;
  g6e.2xlarge) RATE=2.24208 ;;
  *)           RATE=3.0 ;;
esac
KEY=s3://$BUCKET/spend/$(date -u +%Y-%m).txt
TOTAL=$(aws s3 cp "$KEY" - 2>/dev/null || echo 0)
TOTAL=$(echo "$TOTAL + $RATE * 5 / 60" | bc -l)
printf '%.4f\n' "$TOTAL" | aws s3 cp - "$KEY"
if (( $(echo "$TOTAL >= $CAP" | bc -l) )); then
  logger "snow-spend-meter: month total \$$TOTAL >= cap \$$CAP, powering off"
  poweroff
fi
METER

cat > /usr/local/bin/snow-idle-stop <<'IDLE'
#!/bin/bash
# Counts consecutive idle 5-minute checks; 6 in a row (30 min) powers off.
STATE=/var/lib/snow-idle-count
ssh_conns=$(ss -Htn state established '( sport = :22 )' | wc -l)
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1)
if [ "$ssh_conns" -eq 0 ] && [ "${gpu_util:-0}" -lt 5 ]; then
  n=$(( $(cat $STATE 2>/dev/null || echo 0) + 1 ))
else
  n=0
fi
echo $n > $STATE
[ "$n" -ge 6 ] && { logger "snow-idle-stop: idle 30 min, powering off"; echo 0 > $STATE; poweroff; }
IDLE

chmod +x /usr/local/bin/snow-spend-meter /usr/local/bin/snow-idle-stop
echo 0 > /var/lib/snow-idle-count
cat > /etc/cron.d/snow-guards <<'CRON'
*/5 * * * * root /usr/local/bin/snow-spend-meter
*/5 * * * * root /usr/local/bin/snow-idle-stop
CRON
command -v bc >/dev/null || apt-get install -y bc
command -v aws >/dev/null || snap install aws-cli --classic

# NIM: bound to localhost, reached over an SSH tunnel. Restarts on every boot.
NGC_API_KEY=$(aws ssm get-parameter --region us-west-2 --name /app/snow/NVIDIA_API_KEY \
  --with-decryption --query Parameter.Value --output text)
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
# The NIM runs as uid 1000; a Docker-created mount source would be root-owned.
mkdir -p /opt/nim-cache && chown 1000:1000 /opt/nim-cache
docker run -d --name body-pose --restart unless-stopped --gpus all --shm-size=8g \
  -e NGC_API_KEY="$NGC_API_KEY" -v /opt/nim-cache:/opt/nim/.cache \
  -p 127.0.0.1:8000:8000 -p 127.0.0.1:8001:8001 \
  nvcr.io/nim/nvidia/body-pose:latest
