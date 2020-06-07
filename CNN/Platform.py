from player import Player
from random import randrange
from constants import Chessboard

Array = tuple(randrange(720720) for i in range(500))
player_1 = Player(True,Array)
player_2 = Player(False,Array)
Board = Chessboard(Array)

def Get_available(isFirst, Round, mode, board):
    if mode == "position":
        Self_part = board.getNext(isFirst,Round) #自己棋盘落子的位置
        #available为全部可行位置
        available = [Self_part] if Self_part else []
        available += board.getNone(not isFirst)
        return available
    else:
        return [move for move in range(4) if board.copy().move(isFirst, move)]


for Round in range(2000):
    for isFirst in (True,False):
        available = Get_available(isFirst, Round, "position", Board)
        if available:
            
