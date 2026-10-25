from dataclasses import dataclass

@dataclass
class Level0:
    prefix = ""
    suffix = ""
    
@dataclass
class Level1:
    prefix = ""
    suffix = "\nReply with an answer that accounts for the most likely answer the other agent would give."
    
@dataclass
class TeamReasoning:
    prefix = ""
    suffix = "\nPlay like a team."