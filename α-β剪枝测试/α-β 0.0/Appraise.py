from random import randrange
Min_mark = -10000000000000000
Max_Mark = 10000000000000000

def getmark(board):
    '评估函数'
    return randrange(1000000)

if __name__ == "__main__":
    from Source import Chessboard
    c = Chessboard(tuple(randrange(720720) for i in range(500)))
    c.add(True,(0,1))
    c.add(False,(0,4))
    print(c)
    print(getmark(c))
