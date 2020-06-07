from constants import Chessboard
Board = Chessboard(list(range(500)))
import numpy

total = 0
is_First = True
while True:
    Round = total//4

    for _ in range(2):
        A,B = numpy.array(Board.getRaw()).transpose((2,0,1))
        print(A)
        print(B)
        print(Board.getTime(is_First))
        print(Board.getNext(is_First,Round))
        Board.add(is_First,tuple(map(int,input().split())))
        total += 1
        is_First ^= 1
    
    for _ in range(2):
        A,B = numpy.array(Board.getRaw()).transpose((2,0,1))
        print(A)
        print(B)
        print(Board.getTime(is_First))
        Board.move(is_First,int(input()))
        total += 1
        is_First ^= 1
