from .rules import OneGame, TwoOfThree, FirstToGain100, FirstDoble, CapicuaDoble

RULES = [
    OneGame,
    TwoOfThree,
    FirstToGain100,
    FirstDoble,
    CapicuaDoble,
]

def get_rule(value):
    value = value.lower()
    for obj in RULES:
        if obj.__name__.lower() == value:
            return obj
    raise ValueError(f"{value} not found in {[r.__name__ for r in RULES]}")
