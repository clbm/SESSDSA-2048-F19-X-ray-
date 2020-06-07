from random import randrange
Min_mark = -10000000000000000
Max_Mark = 10000000000000000

def getmark(board,isFirst):
    '''
    :param board:棋盘
    :return: 胜利者 True or False
    '''
    ours = board.getScore(isFirst)
    theirs = board.getScore(not isFirst)
    Len_ours = len(ours)
    Len_theirs = len(theirs)
    while Len_ours and Len_theirs:
        O,T = ours.pop(),theirs.pop()
        if O == T:
            Len_ours -= 1
            Len_theirs -= 1
        else:
            return O>T
    return bool(Len_ours)
		
		
		
    

if __name__ == "__main__":
    from Source import Chessboard
    c = Chessboard(tuple(randrange(720720) for i in range(500)))
    c.add(True,(0,1),2)
    c.add(False,(0,4),1)
    c.add(False,(1,4),1)
    c.add(False,(2,4),1)
    c.add(True,(3,1),1)
    print(c)
    print(getmark(c,1))
