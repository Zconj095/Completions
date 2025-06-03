from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import uuid
from random import randint

# --- Stat System ---

class StatType(Enum):
    STRENGTH = "Strength"
    POWER = "Power"
    AGILITY = "Agility"

@dataclass
class Stat:
    type: StatType
    base_value: int

# --- Skill System ---

@dataclass
class Skill:
    name: str
    cost: int
    level: int = 1

    def train(self):
        self.level += 1
        print(f"{self.name} skill leveled up to {self.level}!")

# --- Character System ---

@dataclass
class Character:
    id: uuid.UUID
    name: str
    stats: List[Stat]
    skills: List[Skill] = field(default_factory=list)
    skill_points: int = 5
    level: int = 1
    experience: int = 0
    health: int = 100

    def get_stat(self, stat_type: StatType) -> int:
        for stat in self.stats:
            if stat.type == stat_type:
                return stat.base_value
        return 0

    def add_experience(self, amt: int):
        self.experience += amt
        while self.experience >= self.get_xp_for_level():
            self.experience -= self.get_xp_for_level()
            self.level_up()

    def get_xp_for_level(self) -> int:
        return 100 * (self.level ** 2)

    def level_up(self):
        self.level += 1
        self.skill_points += 2
        self.health += 10
        print(f"{self.name} leveled up to {self.level}! Health: {self.health}, Skill Points: {self.skill_points}")

    def learn_skill(self, skill: Skill):
        if self.skill_points >= skill.cost:
            self.skills.append(skill)
            self.skill_points -= skill.cost
            print(f"{self.name} learned {skill.name}!")
        else:
            print(f"Not enough skill points to learn {skill.name}.")

    def train_skill(self, skill_name: str):
        for skill in self.skills:
            if skill.name == skill_name:
                skill.train()
                return
        print(f"{self.name} does not know {skill_name}.")

    def attack(self, enemy):
        strength = self.get_stat(StatType.STRENGTH)
        dmg = strength + randint(0, 5)
        print(f"{self.name} attacks for {dmg} damage!")
        enemy.take_damage(dmg)

# --- Enemy System ---

@dataclass
class Enemy:
    name: str
    health: int = 100

    def take_damage(self, dmg: int):
        self.health -= dmg
        print(f"{self.name} takes {dmg} damage! (Health: {self.health})")

# --- Example Usage ---

# Create character with stats
maya = Character(
    id=uuid.uuid4(),
    name="Maya",
    stats=[
        Stat(StatType.STRENGTH, 28),
        Stat(StatType.POWER, 24),
        Stat(StatType.AGILITY, 24)
    ]
)

# Learn and train skills
fireball = Skill("Fireball", cost=3)
maya.learn_skill(fireball)
maya.train_skill("Fireball")

# Gain experience and level up
maya.add_experience(500)

# Combat example
orc = Enemy(name="Orc", health=60)
while orc.health > 0:
    maya.attack(orc)
    if orc.health <= 0:
        print(f"{maya.name} defeated the {orc.name}!")
        break

# Show skills
for skill in maya.skills:
    print(f"Skill: {skill.name}, Level: {skill.level}")
