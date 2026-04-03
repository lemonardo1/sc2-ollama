"""Ollama Gemma4를 통한 SC2 전략 결정 모듈"""

import json
import logging
from dataclasses import dataclass

import ollama

from config import OLLAMA_HOST, OLLAMA_MODEL, LLM_TIMEOUT

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """현재 게임 상태를 구조화"""
    race: str
    minerals: int
    vespene: int
    supply_used: int
    supply_cap: int
    worker_count: int
    army_count: int
    army_composition: list[str]
    structures: list[str]
    enemy_race: str
    enemy_units_visible: list[str]
    time_seconds: float
    idle_workers: int
    pending_buildings: list[str]
    tech_available: list[str]
    base_count: int
    gas_buildings: int
    gas_workers: int

    def to_prompt(self) -> str:
        army_str = ', '.join(self.army_composition) if self.army_composition else '없음'
        return f"""현재 SC2 게임 상태:
- 내 종족: {self.race}
- 자원: 미네랄 {self.minerals}, 가스 {self.vespene}
- 서플라이: {self.supply_used}/{self.supply_cap}
- 일꾼 수: {self.worker_count} (유휴: {self.idle_workers}, 가스 일꾼: {self.gas_workers})
- 기지 수: {self.base_count}, 가스 건물 수: {self.gas_buildings}
- 군대 유닛 수: {self.army_count}
- 군대 구성: {army_str}
- 건물: {', '.join(self.structures) if self.structures else '없음'}
- 건설 중: {', '.join(self.pending_buildings) if self.pending_buildings else '없음'}
- 사용 가능 테크: {', '.join(self.tech_available) if self.tech_available else '없음'}
- 적 종족: {self.enemy_race}
- 보이는 적 유닛: {', '.join(self.enemy_units_visible) if self.enemy_units_visible else '없음'}
- 게임 시간: {self.time_seconds:.0f}초"""


SYSTEM_PROMPT = """너는 스타크래프트 2 프로 게이머 AI다. 전략: 3기지 확장 후 테크를 올려 고급 병력을 축적하고 한방에 공격한다.

반드시 아래 형식의 JSON만 반환해라. 설명이나 다른 텍스트는 포함하지 마라.

{
  "reasoning": "현재 상황 분석 (한 줄)",
  "actions": [
    {"type": "build_worker"},
    {"type": "build_supply"},
    {"type": "build_gas"},
    {"type": "build_structure", "name": "건물이름"},
    {"type": "train_unit", "name": "유닛이름"},
    {"type": "attack"},
    {"type": "expand"},
    {"type": "research", "name": "업그레이드이름"},
    {"type": "scout"},
    {"type": "defend"}
  ]
}

전략 가이드라인:
1. 초반(0~180초): 일꾼 16명까지 생산, 가스 건물 1~2개, 1차 확장 준비
2. 중반(180~360초): 2차 확장(3기지 완성), 가스 건물 기지마다 2개씩, 테크 건물 짓기, 일꾼 60~66명
3. 후반(360초~): 고급 유닛 대량 생산, 병력 15~20기 이상 모이면 attack
4. 가스가 부족하면 build_gas를 우선해라
5. 기지당 일꾼 22명(미네랄 16 + 가스 6) 목표, 기지 3개면 일꾼 60~66명
6. 서플라이가 supply_cap - 4 이하로 남으면 build_supply 우선
7. 적이 러시하면 즉시 defend + 기본 유닛 생산으로 전환

종족별 고급 유닛 우선순위:
- 프로토스: Immortal > Colossus > VoidRay > Stalker (Gateway에서 Stalker, Robo에서 Immortal/Colossus, Stargate에서 VoidRay)
  - 테크 경로: Gateway → CyberneticsCore → RoboticsFacility → RoboticsBay (콜로서스용)
- 테란: SiegeTank > Medivac > Marauder > Marine (Factory에서 탱크, Starport에서 메디백)
  - 테크 경로: Barracks → Factory → Starport
- 저그: Hydralisk > Roach > Queen > Zergling (HydraliskDen 필요)
  - 테크 경로: SpawningPool → RoachWarren → HydraliskDen

건물/유닛 이름은 영문 SC2 API 이름을 사용해라:
- 프로토스: Pylon, Gateway, Nexus, CyberneticsCore, RoboticsFacility, RoboticsBay, Stargate, Forge, TwilightCouncil, Stalker, Zealot, Sentry, Immortal, Colossus, VoidRay
- 테란: SupplyDepot, Barracks, CommandCenter, Factory, Starport, Armory, Marine, Marauder, SiegeTank, Medivac, Reaper
- 저그: Overlord, SpawningPool, Hatchery, RoachWarren, HydraliskDen, Zergling, Roach, Hydralisk, Queen
"""


class LLMBrain:
    """Ollama Gemma4 기반 전략 결정 엔진"""

    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_HOST)
        self.model = OLLAMA_MODEL
        self.last_decision = None

    def decide(self, game_state: GameState) -> list[dict]:
        """게임 상태를 분석하고 행동 리스트를 반환"""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": game_state.to_prompt()},
                ],
                options={"temperature": 0.3, "num_predict": 512},
                think=False,
            )

            content = response.message.content.strip()
            # JSON 블록 추출 (```json ... ``` 래핑 처리)
            if "```" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end]

            decision = json.loads(content)
            self.last_decision = decision

            logger.info("LLM 판단: %s", decision.get("reasoning", ""))
            return decision.get("actions", [])

        except json.JSONDecodeError as e:
            logger.warning("LLM 응답 파싱 실패: %s / 원본: %s", e, content[:200])
            return self._fallback_actions(game_state)
        except Exception as e:
            logger.error("LLM 호출 실패: %s", e)
            return self._fallback_actions(game_state)

    def _fallback_actions(self, state: GameState) -> list[dict]:
        """LLM 실패 시 기본 행동 - 3확장 매크로 전략"""
        actions = []
        if state.supply_cap - state.supply_used <= 4:
            actions.append({"type": "build_supply"})
        if state.worker_count < min(22 * state.base_count, 66):
            actions.append({"type": "build_worker"})
        if state.gas_buildings < state.base_count * 2:
            actions.append({"type": "build_gas"})
        if state.base_count < 3 and state.minerals > 400:
            actions.append({"type": "expand"})
        if state.minerals > 300:
            actions.append({"type": "train_unit", "name": "default"})
        return actions
