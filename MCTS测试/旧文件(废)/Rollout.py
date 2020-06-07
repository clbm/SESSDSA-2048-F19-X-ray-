#本程序原为老师给的测试用AI，用于基于一定规律下随机走棋。
#这里将其改写，用于ROLLOUT的模拟


from random import shuffle, randrange
from Appraise import getmark


def simulation(isFirst:bool, currentRound:int, board, mode:int):
    '随机走棋，模拟'
    if mode == 0:  # 给出己方下棋的位置
        another = board.getNext(isFirst, currentRound)  # 己方的允许落子点
        if another != (): return another

        available = board.getNone(not isFirst)  # 对方的允许落子点
        if not available:   # 整个棋盘已满
            return None
        else:
            from random import choice
            return choice(available)
    
    elif mode == 1:  # 给出己方合并的方向
        directionList = [0, 1, 2, 3]
        shuffle(directionList)
        for direction in directionList:
            if board.move(isFirst, direction): return direction
    else:
        return


def Rollout(currentRound:int, board, Gamemode:int, isFirst):
    '模拟对决'
    depth = 0
    max_depth = 5

    while depth < max_depth and currentRound < 500: 

        if Gamemode == 0:
            #先手落子
            position = simulation(True,currentRound,board,0)
            if isinstance(position,tuple):
                board.add(True,position)
            Gamemode = 1
        
        elif Gamemode == 1:
            #后手落子
            position = simulation(False,currentRound,board,0)
            if isinstance(position,tuple):
                board.add(False,position)
            Gamemode = 2
        
        elif Gamemode == 2:
            #先手合并
            direction = simulation(True,currentRound,board.copy(),1)
            if isinstance(direction,int):
                board.move(True,direction)
            Gamemode = 3
        
        elif Gamemode == 3:
            #后手合并
            direction = simulation(False,currentRound,board.copy(),1)
            if isinstance(direction,int):
                board.move(False,direction)
            Gamemode = 0
            depth += 1
            currentRound += 1

    return getmark(board, isFirst)
    

if __name__ == "__main__":
    from Source import Chessboard
    from time import time
    
    Board = Chessboard(tuple(randrange(720720) for i in range(500)))
    t1 = time()
    for _ in range(500):
        Rollout(0, Board, 0,0)
    t2 = time()
    print(t2-t1)