from mas.mas import MetaMAS
from .skill_mas import SkillMAS, GoalGCNMAS

MAS = {
    'skill-mas': SkillMAS,
    'skill': SkillMAS,
    'gcn': SkillMAS,
    # backward compat
    'goal-gcn': SkillMAS,
    'goalrl': SkillMAS,
}

def get_mas(mas_type: str) -> MetaMAS:

    if MAS.get(mas_type) is None:
        raise ValueError(f'Unsupported mas type: {mas_type}. Available: {list(MAS.keys())}')
    return MAS.get(mas_type)()
