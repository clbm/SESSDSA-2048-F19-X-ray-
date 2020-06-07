if __name__ == "__main__":
    from Source import Chessboard #棋盘
from Appraise import getmark,Min_mark,Max_Mark #(伪)评估函数

#Gamemode = 0: 先手落子
#Gamemode = 1: 后手落子
#Gamemode = 2: 先手合并
#Gamemode = 3: 后手合并
#self.mode = 0: 落子
#self.mode = 1: 合并




class Node:
    'min-max树的节点，传入参数分布为:轮数，模式，是否为自己回合，棋盘状态，alpha,beta'

    def __init__(self, Round:int, Gamemode:int, is_max:bool, board, alpha:int = Min_mark, beta:int = Max_Mark):
        '初始化'
        self.Round = Round #轮数
        self.Gamemode = Gamemode #总模式
        self.isFirst = 1^(Gamemode%2) #是否先手
        self.mode = Gamemode // 2 #模式
        self.is_max = is_max #敌我
        self.board = board #棋盘
        self.alpha = alpha #alpha
        self.beta = beta #beta
    



    def simulation(self, depth:int, is_root = False):
        '模拟对战'
        ##################################
            #警告: 还未做防超时保险系统    
        ##################################
        max_depth = 1

        #大于最大深度时:
        if depth >= max_depth or self.Round == 500:
            self.alpha = self.beta = getmark(self.board)
            return


        #以下为一般情况

        if self.mode == 0:   #落子阶段
            #第一步，找出全部可行落子
            Self_part = self.board.getNext(self.isFirst,self.Round) #自己棋盘落子的位置
            #available为全部可行位置
            available = [Self_part] if Self_part else []
            available += self.board.getNone(not self.isFirst) #对手棋盘可落子的位置


            #第二步，迭代模拟

            #无合法移动的特殊情况
            if len(available) == 0:
                child = Node(
                    self.Round,  #局数不变
                    self.Gamemode+1,  #模式+1
                    not self.is_max,  #敌我反转
                    self.board, #新的棋盘
                    self.alpha, #alpha保留
                    self.beta #beta保留
                    )
                child.simulation(depth)
                self.alpha = child.alpha
                self.beta = child.beta
                return

            #一般情况
            for position in available:
                #子树对决
                new_bord = self.board.copy()
                new_bord.add(self.isFirst,position)
                child = Node(
                    self.Round,  #局数不变
                    self.Gamemode+1,  #模式+1
                    not self.is_max,  #敌我反转
                    new_bord, #新的棋盘
                    self.alpha, #alpha保留
                    self.beta #beta保留
                    )
                child.simulation(depth)

                #更新alpha-beta
                if self.is_max:
                    if is_root:
                        #对于树根，需要比较alpha值的变化
                        old_alpha = self.alpha
                    self.alpha = max(self.alpha,child.alpha,child.beta)
                    if is_root and (self.alpha > old_alpha):
                        #当alpha值变大时，落子位置更新
                        result = position
                else:
                    self.beta = min(self.beta,child.alpha,child.beta)
                
                #alpha-beta剪枝
                if self.alpha >= self.beta:
                    break
            
            if is_root:
                return result
            return



        else:   #合并阶段
            No_available = True
            for move in range(4):

                #跳过非法合并
                new_board = self.board.copy()
                if not new_board.move(self.isFirst, move):
                    continue

                #子树对决
                No_available = False
                child = Node(
                    self.Round + (1^self.isFirst),  #局数后手+1
                    (self.Gamemode+1)%4,  #模式+1后对4取余
                    not self.is_max,  #敌我反转
                    new_board, #新的棋盘
                    self.alpha, #alpha保留
                    self.beta #beta保留
                    )
                child.simulation(depth+(1^self.isFirst))

                #更新alpha-beta
                if self.is_max:
                    if is_root:
                        #对于树根，需要比较alpha值的变化
                        old_alpha = self.alpha
                    self.alpha = max(self.alpha,child.alpha,child.beta)
                    if is_root and (self.alpha > old_alpha):
                        #当alpha值变大时，落子位置更新
                        result = move
                else:
                    self.beta = min(self.beta,child.alpha,child.beta)
                
                #alpha-beta剪枝
                if self.alpha >= self.beta:
                    break
                
            if is_root and not No_available:
                return result
            
            #无合法移动的特殊情况
            if No_available:
                child = Node(
                    self.Round + (1^self.isFirst),  #局数后手+1
                    (self.Gamemode+1)%4,  #模式+1后对4取余
                    not self.is_max,  #敌我反转
                    self.board, #新的棋盘
                    self.alpha, #alpha保留
                    self.beta #beta保留
                    )
                child.simulation(depth+(1^self.isFirst))
                self.alpha = child.alpha
                self.beta = child.beta


if __name__ == "__main__":
    from random import randrange
    Board = Chessboard(tuple(randrange(720720) for i in range(500)))
    Board.add(True,Board.getNext(True,0))
    Board.add(False,Board.getNext(False,0))
    Board.move(True,1)
    print(Board)
    node = Node(0,3,True,Board)
    print(node.simulation(0,True))
