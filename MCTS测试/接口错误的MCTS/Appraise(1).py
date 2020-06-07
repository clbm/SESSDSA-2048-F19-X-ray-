from random import randrange
Min_mark = -10000000000000000
Max_Mark = 10000000000000000

def getmark(board,isFirst):
    '评估函数'
    #part1:棋子直接估值部分
    #获得两方棋子数值的从大到小的列表
    ours = board.getScore(isFirst)
    print(ours)
    their = board.getScore(not isFirst)
    ours.reverse()
    their.reverse()
    i1 = 0
    while ours and their and ours[0] >= their[0]:   #判断两方的最大值，第二大值......
        i1 += 1   
        ours.pop(0)
        their.pop(0)
    return i1

    #part2:优先下在自己的领地中 
		
		
		
    

if __name__ == "__main__":
    from Source import Chessboard
    c = Chessboard(tuple(randrange(720720) for i in range(500)))
    c.add(True,(0,1))
    c.add(True,(1,2),2)
    c.add(True,(2,2),2)
    c.add(True,(3,2),4)
    c.add(False,(0,4))
    print(c)
    print(getmark(c,1))
