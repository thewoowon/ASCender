#!/bin/bash
set -e
apt-get update -y
apt-get install -y tmux awscli jq

# Conda env (DLAMI엔 이미 conda 있음)
eval "$(/home/ubuntu/anaconda3/bin/conda shell.bash hook)" || true
conda create -y -n asc python=3.10
conda activate asc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install datasets sacrebleu sentencepiece accelerate wandb

# 프로젝트 배치
mkdir -p /home/ubuntu/asc && cd /home/ubuntu/asc
aws s3 sync s3://$${BUCKET:-REPLACE_ME}/code ./code || true
aws s3 sync s3://$${BUCKET:-REPLACE_ME}/data ./data || true

# 스팟 중단 알림 감시 → 즉시 체크포인트 업로드
cat >/usr/local/bin/asc-spot-guard.sh <<'EOF'
#!/bin/bash
BUCKET="${BUCKET}"
CKPT_DIR="/home/ubuntu/asc/outputs/checkpoints"
while true; do
  if curl -s http://169.254.169.254/latest/meta-data/spot/instance-action | grep -q "termination-time"; then
    echo "[SpotGuard] termination notice received, uploading checkpoints..."
    aws s3 sync "$CKPT_DIR" "s3://$${BUCKET}/checkpoints/$(hostname)/" || true
    sync
    sleep 70
  fi
  sleep 5
done
EOF
chmod +x /usr/local/bin/asc-spot-guard.sh
cat >/etc/systemd/system/asc-spot-guard.service <<'EOF'
[Unit]
Description=ASC Spot Interruption Guard
After=network.target
[Service]
Type=simple
Environment=BUCKET=REPLACE_ME
ExecStart=/usr/local/bin/asc-spot-guard.sh
Restart=always
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload && systemctl enable --now asc-spot-guard

# 학습 디렉토리
mkdir -p /home/ubuntu/asc/outputs/checkpoints
chown -R ubuntu:ubuntu /home/ubuntu
echo "GPU node ready."
