import numpy as np
from Source import Chessboard
Weight = {True:np.array((1.0,1.05,1.1,1.15,1.2,1.2,1.2,1.2)),False:np.array((1.2,1.2,1.2,1.2,1.15,1.1,1.05,1.0))}


def getmark(board:Chessboard, isFirst:bool):
    "评估函数"
    Value, Belong = np.array(board.getRaw(),).transpose((2,0,1))
    Belong ^= isFirst
    Ours = Belong<<Value
    Theirs = (1-Belong)<<Value
    return np.sum((Ours*Weight[isFirst]-Theirs*Weight[not isFirst])*(1+0.02*Value))
