from Structure import Node
from MCTS import MCTS

class Player:
    '最终的接口'
    def __init__(self, isFirst:bool, array:tuple):
        '初始化'
        self.isFirst = isFirst
        self.begin = True
    
    def output(self, currentRound: int, board, mode: str):
        '输出结果'
        No_way =  mode[0] == '_'
        if 'position' in mode:
            mode = 0
        else:
            mode = 1
        
        #新的树根
        if self.begin:
            self.root = Node(1-self.isFirst, currentRound, board)
            self.begin = False
        else:
            self.root = self.root.get_newroot(board)
            if self.root:
                self.root.parent = None
            else:
                self.root = Node(1-self.isFirst, currentRound, board)

        #开始搜索
        if No_way:
            self.root = self.root.Nextchild(0)
            return None
        else:
            self.root, result = MCTS(self.root, mode*2+1-self.isFirst, currentRound)
            return result


if __name__ == "__main__":
    from Source import Chessboard
    from random import randrange
    Array = tuple(randrange(720720) for i in range(500))
    player1 = Player(True, Array)
    player2 = Player(False, Array)
    board = Chessboard(Array)

    for r in range(10):
        operation = player1.output(r, board, 'position')
        print(operation)
        board.add(True, operation)

        operation = player2.output(r, board, 'position')
        print(operation)
        board.add(True, operation)

        operation = player1.output(r, board, 'direction')
        print(operation)
        board.move(True, operation)

        operation = player2.output(r, board, 'direction')
        print(operation)
        board.move(True, operation)