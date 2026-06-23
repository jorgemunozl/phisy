import numpy as np

class Position():
    """
    A numpy object. dont forget.
    """
    POSX : float
    POSY : float
 

def distance(pos1, pos2) -> float:
    difx = pos1.x - pos2.x
    dify = pos1.y -pos2.y
    return np.sqrt(difx**2 + dify**2)


def magnetic_field_moduli(pos: Position , pos_eval: Position)-> float:
    """
    It takes a wire and return the modu
    """
    return 1/distance(pos,pos_eval)


def magnetic_field_direction():
    pass
def main():
    return 