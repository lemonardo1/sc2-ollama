"""SC2-Ollama 봇 설정"""

# Ollama 설정
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma4"

# SC2 설정
SC2_MAP = "AcropolisLE"
SC2_REALTIME = False

# 봇 설정
# LLM 호출 주기 (게임 스텝 기준, 낮을수록 자주 호출)
LLM_CALL_INTERVAL = 100

# LLM 응답 타임아웃 (초)
LLM_TIMEOUT = 10.0
