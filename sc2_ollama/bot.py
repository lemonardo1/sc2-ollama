"""Ollama Gemma4 기반 SC2 봇 코어 - 3확장 매크로 전략"""

import logging
from collections import Counter

from sc2.bot_ai import BotAI
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2

from config import LLM_CALL_INTERVAL
from sc2_ollama.llm_brain import LLMBrain, GameState

logger = logging.getLogger(__name__)


# 종족별 가스 건물
GAS_BUILDING = {
    Race.Protoss: UnitTypeId.ASSIMILATOR,
    Race.Terran: UnitTypeId.REFINERY,
    Race.Zerg: UnitTypeId.EXTRACTOR,
}

# 종족별 유닛/건물 매핑
RACE_CONFIG = {
    Race.Protoss: {
        "worker": UnitTypeId.PROBE,
        "supply": UnitTypeId.PYLON,
        "townhall": UnitTypeId.NEXUS,
        "townhall_types": {UnitTypeId.NEXUS},
        "basic_military": UnitTypeId.STALKER,
        "structures": {
            "Gateway": UnitTypeId.GATEWAY,
            "CyberneticsCore": UnitTypeId.CYBERNETICSCORE,
            "RoboticsFacility": UnitTypeId.ROBOTICSFACILITY,
            "RoboticsBay": UnitTypeId.ROBOTICSBAY,
            "Stargate": UnitTypeId.STARGATE,
            "Forge": UnitTypeId.FORGE,
            "TwilightCouncil": UnitTypeId.TWILIGHTCOUNCIL,
            "Pylon": UnitTypeId.PYLON,
            "Nexus": UnitTypeId.NEXUS,
        },
        "units": {
            "Zealot": UnitTypeId.ZEALOT,
            "Stalker": UnitTypeId.STALKER,
            "Immortal": UnitTypeId.IMMORTAL,
            "Colossus": UnitTypeId.COLOSSUS,
            "VoidRay": UnitTypeId.VOIDRAY,
            "Sentry": UnitTypeId.SENTRY,
            "default": UnitTypeId.STALKER,
        },
    },
    Race.Terran: {
        "worker": UnitTypeId.SCV,
        "supply": UnitTypeId.SUPPLYDEPOT,
        "townhall": UnitTypeId.COMMANDCENTER,
        "townhall_types": {UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS},
        "basic_military": UnitTypeId.MARINE,
        "structures": {
            "Barracks": UnitTypeId.BARRACKS,
            "Factory": UnitTypeId.FACTORY,
            "Starport": UnitTypeId.STARPORT,
            "Armory": UnitTypeId.ARMORY,
            "EngineeringBay": UnitTypeId.ENGINEERINGBAY,
            "SupplyDepot": UnitTypeId.SUPPLYDEPOT,
            "CommandCenter": UnitTypeId.COMMANDCENTER,
        },
        "units": {
            "Marine": UnitTypeId.MARINE,
            "Marauder": UnitTypeId.MARAUDER,
            "SiegeTank": UnitTypeId.SIEGETANK,
            "Medivac": UnitTypeId.MEDIVAC,
            "Reaper": UnitTypeId.REAPER,
            "default": UnitTypeId.MARINE,
        },
    },
    Race.Zerg: {
        "worker": UnitTypeId.DRONE,
        "supply": UnitTypeId.OVERLORD,
        "townhall": UnitTypeId.HATCHERY,
        "townhall_types": {UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE},
        "basic_military": UnitTypeId.ROACH,
        "structures": {
            "SpawningPool": UnitTypeId.SPAWNINGPOOL,
            "RoachWarren": UnitTypeId.ROACHWARREN,
            "HydraliskDen": UnitTypeId.HYDRALISKDEN,
            "BanelingNest": UnitTypeId.BANELINGNEST,
            "Hatchery": UnitTypeId.HATCHERY,
        },
        "units": {
            "Zergling": UnitTypeId.ZERGLING,
            "Roach": UnitTypeId.ROACH,
            "Hydralisk": UnitTypeId.HYDRALISK,
            "Queen": UnitTypeId.QUEEN,
            "default": UnitTypeId.ROACH,
        },
    },
}

# 연구 매핑
RESEARCH_MAP = {
    "WarpGate": (AbilityId.RESEARCH_WARPGATE, UnitTypeId.CYBERNETICSCORE),
    "Charge": (AbilityId.RESEARCH_CHARGE, UnitTypeId.TWILIGHTCOUNCIL),
    "Blink": (AbilityId.RESEARCH_BLINK, UnitTypeId.TWILIGHTCOUNCIL),
    "ExtendedThermalLance": (AbilityId.RESEARCH_EXTENDEDTHERMALLANCE, UnitTypeId.ROBOTICSBAY),
    "CombatShield": (AbilityId.RESEARCH_COMBATSHIELD, UnitTypeId.BARRACKSTECHLAB),
    "Stimpack": (AbilityId.BARRACKSTECHLABRESEARCH_STIMPACK, UnitTypeId.BARRACKSTECHLAB),
    "GlialReconstitution": (AbilityId.RESEARCH_GLIALREGENERATION, UnitTypeId.ROACHWARREN),
    "GroovedSpines": (AbilityId.RESEARCH_GROOVEDSPINES, UnitTypeId.HYDRALISKDEN),
}


class OllamaBot(BotAI):
    """Ollama Gemma4로 전략을 결정하는 SC2 봇 - 3확장 매크로"""

    def __init__(self):
        super().__init__()
        self.brain = LLMBrain()
        self.pending_actions: list[dict] = []
        self.last_llm_step = -999  # 첫 스텝에서 즉시 호출
        self.attacking = False  # 공격 모드 유지

    @property
    def cfg(self) -> dict:
        return RACE_CONFIG[self.race]

    async def on_start(self):
        logger.info("봇 시작 - 종족: %s, 맵: %s", self.race, self.game_info.map_name)

    async def on_step(self, iteration: int):
        # 항상 수���: 유��� 일꾼 배정 + ��스 일꾼 배���
        await self._manage_workers()

        # 자동 방어: 본진 근처 적 감지 시 군대 긴급 방어
        await self._auto_defend()

        # 공격 모드: 새로 생산된 유닛도 자동 합류
        if self.attacking:
            await self._reinforce_attack()

        # 주��적으로 LLM에 전략 질의
        if iteration - self.last_llm_step >= LLM_CALL_INTERVAL or not self.pending_actions:
            self.pending_actions = self.brain.decide(self._build_game_state())
            self.last_llm_step = iteration
            logger.info("[Step %d] LLM actions: %s", iteration, self.pending_actions)

        # 액션 실행
        remaining = []
        for action in self.pending_actions:
            success = await self._execute_action(action)
            if not success:
                remaining.append(action)
        self.pending_actions = remaining

    async def _reinforce_attack(self):
        """공격 모드일 때 유휴 군대 유닛을 전선으로 보냄"""
        army = self.units.filter(lambda u: u.can_attack and u.type_id != self.cfg["worker"])
        idle_army = army.idle
        if not idle_army:
            # 군대가 전멸하면 공격 모드 해제
            if army.amount < 3:
                self.attacking = False
                logger.info("군대 전멸 - 공격 모드 해제, 재축적 시작")
            return

        target = self.enemy_start_locations[0] if self.enemy_start_locations else self.game_info.map_center
        if self.enemy_structures:
            target = self.enemy_structures.closest_to(self.start_location).position
        for unit in idle_army:
            unit.attack(target)

    async def _auto_defend(self):
        """본진/확장 근처에 적이 들어오면 자동으로 군대 방어"""
        for th in self.townhalls:
            enemies_near = self.enemy_units.closer_than(25, th)
            if enemies_near.amount > 0:
                army = self.units.filter(lambda u: u.can_attack and u.type_id != self.cfg["worker"])
                if army:
                    target = enemies_near.closest_to(th).position
                    for unit in army.idle | army.filter(lambda u: u.is_moving and not u.is_attacking):
                        unit.attack(target)
                break

    def _build_game_state(self) -> GameState:
        """현재 게임 정보를 GameState로 변환"""
        structures = [s.name for s in self.structures]
        pending = [s.name for s in self.structures.filter(lambda s: not s.is_ready)]
        enemy_units = [u.name for u in self.enemy_units]

        # 군대 구성
        army = self.units.filter(lambda u: u.can_attack and u.type_id != self.cfg["worker"])
        army_counter = Counter(u.name for u in army)
        army_composition = [f"{name}x{count}" for name, count in army_counter.most_common()]

        # 사용 가능한 테크 목록
        tech = []
        for name, uid in self.cfg["structures"].items():
            if self.structures(uid).ready.exists:
                tech.append(name)

        # 가스 건물 수, 가스 일꾼 수
        gas_type = GAS_BUILDING[self.race]
        gas_buildings = self.structures(gas_type).ready.amount
        gas_workers = sum(
            geyser.assigned_harvesters
            for geyser in self.gas_buildings
            if geyser.is_ready
        )

        return GameState(
            race=self.race.name,
            minerals=self.minerals,
            vespene=self.vespene,
            supply_used=self.supply_used,
            supply_cap=self.supply_cap,
            worker_count=self.workers.amount,
            army_count=army.amount,
            army_composition=army_composition,
            structures=structures,
            enemy_race=self.enemy_race.name if self.enemy_race else "Unknown",
            enemy_units_visible=enemy_units,
            time_seconds=self.time,
            idle_workers=self.workers.idle.amount,
            pending_buildings=pending,
            tech_available=tech,
            base_count=self.townhalls.amount,
            gas_buildings=gas_buildings,
            gas_workers=gas_workers,
        )

    async def _manage_workers(self):
        """일꾼 관리: 유휴 일꾼 배정 + 가스 일꾼 분배"""
        # 가스 건물에 일꾼 3명씩 배정
        for gas in self.gas_buildings:
            if not gas.is_ready:
                continue
            if gas.assigned_harvesters < gas.ideal_harvesters:
                needed = gas.ideal_harvesters - gas.assigned_harvesters
                workers = self.workers.closer_than(20, gas)
                if not workers:
                    workers = self.workers
                for worker in workers.idle[:needed] if workers.idle.amount >= needed else workers.gathering[:needed]:
                    worker.gather(gas)

        # 남은 유휴 일꾼은 미네랄로
        for worker in self.workers.idle:
            if self.mineral_field:
                mf = self.mineral_field.closest_to(worker)
                worker.gather(mf)

    async def _execute_action(self, action: dict) -> bool:
        """단일 행동을 실행. 성공하면 True 반환."""
        action_type = action.get("type", "")

        try:
            if action_type == "build_worker":
                return await self._build_worker()
            elif action_type == "build_supply":
                return await self._build_supply()
            elif action_type == "build_gas":
                return await self._build_gas()
            elif action_type == "build_structure":
                return await self._build_structure(action.get("name", ""))
            elif action_type == "train_unit":
                return await self._train_unit(action.get("name", "default"))
            elif action_type == "attack":
                return await self._attack()
            elif action_type == "expand":
                return await self._expand()
            elif action_type == "scout":
                return await self._scout()
            elif action_type == "defend":
                return await self._defend()
            elif action_type == "research":
                return await self._research(action.get("name", ""))
            else:
                logger.warning("알 수 없는 행동: %s", action_type)
                return True
        except Exception as e:
            logger.error("행동 실행 실패 [%s]: %s", action_type, e)
            return True  # 에러 시 재시도 방지

    async def _build_worker(self) -> bool:
        # 기지당 22명 한도, 최대 66명
        max_workers = min(22 * self.townhalls.amount, 66)
        if self.workers.amount >= max_workers:
            return True

        # 포화 안 된 기지에서 우선 생산
        for th in self.townhalls.ready.idle:
            if th.assigned_harvesters < th.ideal_harvesters and self.can_afford(self.cfg["worker"]):
                th.train(self.cfg["worker"])
                return True
        # 포화됐어도 총 수가 부족하면 아무 기지에서
        townhalls = self.townhalls.ready.idle
        if townhalls and self.can_afford(self.cfg["worker"]):
            townhalls.random.train(self.cfg["worker"])
            return True
        return False

    async def _build_supply(self) -> bool:
        if self.supply_cap >= 200:
            return True

        if self.race == Race.Zerg:
            if self.can_afford(UnitTypeId.OVERLORD):
                larvae = self.larva
                if larvae:
                    larvae.random.train(UnitTypeId.OVERLORD)
                    return True
            return False

        supply_type = self.cfg["supply"]
        # 여러 개 동시 건설 허용 (서플라이 차이에 비례)
        pending = self.already_pending(supply_type)
        supply_left = self.supply_cap - self.supply_used
        if pending > 0 and supply_left > 2:
            return True
        if self.can_afford(supply_type) and self.townhalls.ready.exists:
            await self.build(supply_type, near=self.townhalls.ready.random.position.towards(self.game_info.map_center, 5))
            return True
        return False

    async def _build_gas(self) -> bool:
        """가스 건물 건설 (기지 근처 빈 가스 간헐천에)"""
        gas_type = GAS_BUILDING[self.race]

        for townhall in self.townhalls.ready:
            # 이 기지 근처 간헐천 중 아직 가스 건물이 없는 것
            geysers = self.vespene_geyser.closer_than(10, townhall)
            for geyser in geysers:
                if not self.can_afford(gas_type):
                    return False
                # 이미 건물이 있는지 확인
                if self.gas_buildings.closer_than(1, geyser).exists:
                    continue
                if self.already_pending(gas_type) >= 1:
                    return True
                worker = self.workers.closest_to(geyser)
                if worker:
                    worker.build_gas(geyser)
                    return True
        return True  # 모든 간헐천에 건물이 있음

    async def _build_structure(self, name: str) -> bool:
        structure_id = self.cfg["structures"].get(name)
        if not structure_id:
            logger.warning("알 수 없는 건물: %s", name)
            return True

        # Gateway는 여러 개 허용 (기지 수에 비례)
        multi_allowed = {UnitTypeId.GATEWAY, UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT}
        if structure_id in multi_allowed:
            max_count = self.townhalls.amount * 2
            if self.structures(structure_id).amount + self.already_pending(structure_id) >= max_count:
                return True
        else:
            if self.structures(structure_id).exists or self.already_pending(structure_id) > 0:
                return True

        if not self.can_afford(structure_id):
            return False

        if not self.townhalls.ready.exists:
            return False

        if self.race == Race.Zerg:
            if self.workers.exists:
                pos = await self.find_placement(structure_id, near=self.townhalls.ready.random.position)
                if pos:
                    worker = self.workers.closest_to(pos)
                    worker.build(structure_id, pos)
                    return True
        else:
            # 파일런/서플라이가 있어야 건물 건설 가능 (프토/테란)
            if self.race == Race.Protoss and structure_id != UnitTypeId.PYLON and structure_id != UnitTypeId.NEXUS:
                if not self.structures(UnitTypeId.PYLON).ready.exists:
                    return False
            pos = await self.find_placement(structure_id, near=self.townhalls.ready.random.position.towards(self.game_info.map_center, 5))
            if pos:
                await self.build(structure_id, near=pos)
                return True
        return False

    async def _train_unit(self, name: str) -> bool:
        unit_id = self.cfg["units"].get(name, self.cfg["basic_military"])
        if not self.can_afford(unit_id):
            return False
        if self.supply_left < 2:
            return False

        if self.race == Race.Zerg:
            larvae = self.larva
            if larvae:
                larvae.random.train(unit_id)
                return True
            return False

        # 생산 건물 찾기
        production_map = {
            # Protoss
            UnitTypeId.ZEALOT: UnitTypeId.GATEWAY,
            UnitTypeId.STALKER: UnitTypeId.GATEWAY,
            UnitTypeId.SENTRY: UnitTypeId.GATEWAY,
            UnitTypeId.IMMORTAL: UnitTypeId.ROBOTICSFACILITY,
            UnitTypeId.COLOSSUS: UnitTypeId.ROBOTICSFACILITY,
            UnitTypeId.VOIDRAY: UnitTypeId.STARGATE,
            # Terran
            UnitTypeId.MARINE: UnitTypeId.BARRACKS,
            UnitTypeId.MARAUDER: UnitTypeId.BARRACKS,
            UnitTypeId.REAPER: UnitTypeId.BARRACKS,
            UnitTypeId.SIEGETANK: UnitTypeId.FACTORY,
            UnitTypeId.MEDIVAC: UnitTypeId.STARPORT,
        }

        prod_building = production_map.get(unit_id)
        if not prod_building:
            return True

        buildings = self.structures(prod_building).ready.idle
        if buildings:
            buildings.random.train(unit_id)
            return True
        return False

    async def _attack(self) -> bool:
        army = self.units.filter(lambda u: u.can_attack and u.type_id != self.cfg["worker"])
        army_supply = sum(self._unit_supply(u) for u in army)

        # 첫 공격: 10기 이상 or 서플라이 40 이상
        if not self.attacking:
            if army.amount < 10 and army_supply < 40:
                logger.info("공격 보류 - 병력 %d (서플라이 %d)", army.amount, army_supply)
                return True
            self.attacking = True

        # 공격 모드 시작 후에는 새로 생산된 유닛도 계속 합류
        target = self.enemy_start_locations[0] if self.enemy_start_locations else self.game_info.map_center
        if self.enemy_structures:
            target = self.enemy_structures.closest_to(self.start_location).position
        elif self.enemy_units:
            target = self.enemy_units.closest_to(self.start_location).position

        for unit in army:
            unit.attack(target)
        logger.info("공격중! 병력 %d (서플라이 %d) -> %s", army.amount, army_supply, target)
        return True

    @staticmethod
    def _unit_supply(unit) -> int:
        """유닛의 서플라이 비용 (대략)"""
        heavy = {UnitTypeId.COLOSSUS, UnitTypeId.SIEGETANK, UnitTypeId.ULTRALISK}
        medium = {UnitTypeId.IMMORTAL, UnitTypeId.STALKER, UnitTypeId.MARAUDER, UnitTypeId.ROACH, UnitTypeId.HYDRALISK, UnitTypeId.VOIDRAY, UnitTypeId.MEDIVAC}
        if unit.type_id in heavy:
            return 6
        if unit.type_id in medium:
            return 3
        return 2

    async def _expand(self) -> bool:
        if self.townhalls.amount >= 4:
            return True  # 4기지 이상은 안 함
        townhall_type = self.cfg["townhall"]
        if not self.can_afford(townhall_type):
            return False
        try:
            await self.expand_now()
            logger.info("확장 시도! (현재 %d기지)", self.townhalls.amount)
            return True
        except Exception as e:
            logger.warning("확장 실패: %s", e)
            return False

    async def _scout(self) -> bool:
        workers = self.workers
        if not workers:
            return True
        scout = workers.random
        target = self.enemy_start_locations[0] if self.enemy_start_locations else self.game_info.map_center
        scout.move(target)
        return True

    async def _defend(self) -> bool:
        army = self.units.filter(lambda u: u.can_attack and u.type_id != self.cfg["worker"])
        if not army:
            return True
        rally = self.townhalls.ready.random.position if self.townhalls.ready else self.start_location
        for unit in army:
            unit.attack(rally)
        return True

    async def _research(self, name: str) -> bool:
        entry = RESEARCH_MAP.get(name)
        if not entry:
            logger.info("알 수 없는 연구: %s", name)
            return True
        ability, building_type = entry
        buildings = self.structures(building_type).ready.idle
        if buildings:
            buildings.random(ability)
            logger.info("연구 시작: %s", name)
            return True
        return False
