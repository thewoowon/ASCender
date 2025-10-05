#!/bin/bash
set -e
apt-get update -y
apt-get install -y awscli jq tmux python3-pip
pip3 install boto3

# 간단한 런처: 대기 중인 GPU 인스턴스 찾고 SSM or SSH로 학습 스크립트 실행(선호 방식 택1)
echo "Orchestrator ready."
