#本程序建立基本结构

from math import log,sqrt


def UCB1(node,N) -> float:
    '返回函数值'
    return node.q/node.n+2*sqrt(log(N)/node.n)


class Node:
    '蒙特卡洛搜索树的节点'
    
    def __init__(self, Gamemode:int, currentRound:int, board, parent=None):
        '初始化'
        self.Gamemode = Gamemode #模式
        self.board = board #棋盘
        self.childs = [] #已探索的子节点
        self.parent = parent #父节点
        self.Round = currentRound #当前局数
        available = [] #下一步可行的操作
        isFirst = 1-Gamemode%2 #是否先手
        self.q = 0 #q值
        self.n = 0 #访问次数

        #最底层不用继续扩展
        if currentRound == 500:
            self.available = iter([])
            self.operations = []
            return
        
        #得到可行的所有点
        if Gamemode // 2:
            #合并阶段
            for direction in range(4):
                if board.copy().move(isFirst, direction):
                    available.append(direction)
        else:
            #落子阶段
            available = []
            self_pos = board.getNext(isFirst, currentRound)
            if self_pos:
                available.append(self_pos)
            available += board.getNone(not isFirst) #自己的加对手的
        self.available = iter(available)
        self.operations = available.copy()
    

    def Get_operation(self):
        '返回已探索的最大点对应的操作'
        MAX_mark = -1000000000
        index = 0
        result = 0
        for child in self.childs:
            mark = UCB1(child,self.n)
            if  mark > MAX_mark:
                result = index
                MAX_mark = mark
            index += 1
        return self.childs[result], self.operations[result]


    def __str__(self) -> str:
        '返回棋盘的字符串，方便比对'
        return self.board.__str__()


    def get_newroot(self, board):
        '返回下一步的节点，以继承搜索过的树'
        aim = str(board)
        for child in self.childs:
            if str(child) == aim:
                return child
    

    def Nextchild(self,N) -> tuple:
        '返回UCB1值最大的节点'
        if self.available.__length_hint__():
            #有未探索的节点
            operation = next(self.available)
            isFirst = 1-self.Gamemode%2
            Round_end = self.Gamemode == 4
            new_board = self.board.copy()
            #添加新节点
            if self.Gamemode //2:
                new_board.move(isFirst,operation)
            else:
                new_board.add(isFirst,operation)
            
            new_node = Node(
                Gamemode = self.Gamemode+1-4*Round_end, 
                currentRound = self.Round+Round_end,
                board = new_board,
                parent= self
                    )
            
            self.childs.append(new_node)
            return new_node, False
        
        elif not len(self.childs):
            Round_end = self.Gamemode == 4
            new_node = Node(
                Gamemode = self.Gamemode+1-4*Round_end, 
                currentRound = self.Round+Round_end,
                board = self.board.copy(),
                parent= self
                    )
            self.childs.append(new_node)
            return new_node, False
            
        
        else:
            return max(self.childs,key=lambda x:UCB1(x,N)), True





if __name__ == "__main__":
    from Source import Chessboard
    from random import randrange
    Board = Chessboard(tuple(randrange(720720) for i in range(500)))
    node = Node(0,0,Board)
    print(node)
    for _ in range(3):
        print('=====================')
        print(node.Nextchild(0)[0])
    