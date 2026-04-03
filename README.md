# SC2-Ollama Bot

Ollama Gemma4 모델로 실시간 전략을 결정하는 StarCraft II 자동 플레이 봇.

3기지 확장 → 테크업 → 고급 병력 축적 → 총공격 매크로 전략을 사용한다.

## 사전 요구사항

- **StarCraft II** 설치 (Battle.net)
- **Ollama** 설치 및 Gemma4 모델 다운로드
- **Python 3.11+**
- SC2 Maps 디렉토리에 래더맵 필요 (`/Applications/StarCraft II/maps/`)

## 설치

```bash
# Ollama에 Gemma4 모델 준비
ollama pull gemma4

# Python 의존성 설치
pip install -r requirements.txt
```

## 실행

```bash
# 기본 실행 (랜덤 종족, 미디엄 난이도)
python main.py

# 프로토스로 하드 난이도
python main.py --race protoss --difficulty hard

# 저그 vs 테란, 실시간 모드
python main.py --race zerg --enemy-race terran --realtime

# 특정 맵 지정
python main.py --map "AcropolisLE"
```

## 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--race` | 봇 종족 (protoss/terran/zerg/random) | random |
| `--enemy-race` | 적 종족 | random |
| `--difficulty` | AI 난이도 (veryeasy~cheatinsane) | medium |
| `--map` | 맵 이름 | AcropolisLE |
| `--realtime` | 실시간 모드 | off |

## 구조

```
sc2-ollama/
├── main.py                  # 진입점 (argparse CLI)
├── config.py                # Ollama/SC2/봇 설정
├── requirements.txt         # burnysc2 + ollama
└── sc2_ollama/
    ├── __init__.py
    ├── bot.py               # SC2 봇 코어 - 3확장 매크로 전략
    └── llm_brain.py         # Ollama Gemma4 전략 결정 모듈
```

## 전략

1. **초반 (0~180초)**: 일꾼 16명 생산, 가스 건물 건설, 1차 확장
2. **중반 (180~360초)**: 3기지 완성, 기지당 가스 2개, 테크 건물 건설, 일꾼 60~66명
3. **후반 (360초~)**: 고급 유닛 대량 생산 (Immortal, Colossus, VoidRay 등), 병력 축적 후 총공격

## 동작 방식

1. `bot.py`가 매 게임 스텝마다 게임 상태(자원, 유닛, 건물, 군대 구성 등)를 수집
2. 일정 주기(`LLM_CALL_INTERVAL`)마다 `llm_brain.py`가 Gemma4에 상태를 전달
3. Gemma4가 JSON으로 행동 리스트(건물, 유닛 생산, 가스, 확장, 공격 등)를 반환
4. 봇이 행동을 순서대로 실행 + 자동 방어/가스 일꾼 배분/공격 합류 처리
5. LLM 호출 실패 시 3확장 매크로 폴백 전략으로 동작

## 종족별 테크 경로

- **프로토스**: Gateway → CyberneticsCore → RoboticsFacility → RoboticsBay (Immortal/Colossus)
- **테란**: Barracks → Factory → Starport (SiegeTank/Medivac)
- **저그**: SpawningPool → RoachWarren → HydraliskDen (Roach/Hydralisk)
