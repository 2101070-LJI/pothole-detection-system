FROM cloudflare/cloudflared:latest

# 설정 파일 복사
COPY config.yml /etc/cloudflared/config.yml

# 자동 터널 실행 (credentials 파일에서 tunnel ID 자동 감지)
ENTRYPOINT ["cloudflared", "tunnel", "--config", "/etc/cloudflared/config.yml", "run"]

