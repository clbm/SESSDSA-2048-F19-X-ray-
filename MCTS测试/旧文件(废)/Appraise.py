from random import randrange
Min_mark = -1
Max_Mark = 1<<200

def getmark(board, isFirst:bool):
    '评估函数'
    #part1:棋子直接估值部分
    #获得两方棋子数值的从大到小的列表
    ours = board.getScore(isFirst)
    theirs = board.getScore(not isFirst)
    zeros = 32-len(ours)-len(theirs)
    result = zeros*(1<<70)
    for chess in ours:
        result += 1<<(14+chess)*5
    for chess in theirs:
        result += 1<<(14-chess)*5
    return result


if __name__ == "__main__":
    from Source import Chessboard
    c = Chessboard(tuple(randrange(720720) for i in range(500)))
    c.add(True,(0,1),1)
    c.add(True,(0,3),2)
    c.add(True,(0,2))
    c.add(False,(0,4))
    print(c)
    print(getmark(c,1))
