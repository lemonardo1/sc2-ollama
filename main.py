"""SC2-Ollama 봇 실행 진입점"""

import argparse
import logging
import sys

from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

from sc2_ollama.bot import OllamaBot
from config import SC2_MAP, SC2_REALTIME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("sc2-ollama")

RACE_MAP = {
    "protoss": Race.Protoss,
    "terran": Race.Terran,
    "zerg": Race.Zerg,
    "random": Race.Random,
}

DIFFICULTY_MAP = {
    "veryeasy": Difficulty.VeryEasy,
    "easy": Difficulty.Easy,
    "medium": Difficulty.Medium,
    "mediumhard": Difficulty.MediumHard,
    "hard": Difficulty.Hard,
    "harder": Difficulty.Harder,
    "veryhard": Difficulty.VeryHard,
    "cheatvision": Difficulty.CheatVision,
    "cheatmoney": Difficulty.CheatMoney,
    "cheatinsane": Difficulty.CheatInsane,
}


def main():
    parser = argparse.ArgumentParser(description="SC2 Ollama Gemma4 Bot")
    parser.add_argument("--race", choices=RACE_MAP.keys(), default="random", help="봇 종족 (default: random)")
    parser.add_argument("--enemy-race", choices=RACE_MAP.keys(), default="random", help="적 종족 (default: random)")
    parser.add_argument("--difficulty", choices=DIFFICULTY_MAP.keys(), default="medium", help="적 난이도 (default: medium)")
    parser.add_argument("--map", default=SC2_MAP, help=f"맵 이름 (default: {SC2_MAP})")
    parser.add_argument("--realtime", action="store_true", default=SC2_REALTIME, help="실시간 모드")
    args = parser.parse_args()

    bot_race = RACE_MAP[args.race]
    enemy_race = RACE_MAP[args.enemy_race]
    difficulty = DIFFICULTY_MAP[args.difficulty]

    logger.info("=== SC2 Ollama Bot ===")
    logger.info("봇 종족: %s | 적: %s %s | 맵: %s", args.race, args.enemy_race, args.difficulty, args.map)

    run_game(
        maps.get(args.map),
        [
            Bot(bot_race, OllamaBot()),
            Computer(enemy_race, difficulty),
        ],
        realtime=args.realtime,
    )


if __name__ == "__main__":
    main()
