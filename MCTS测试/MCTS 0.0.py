#本程序为蒙特卡洛树的实现
#接口为评估函数与随机落子函数
from numpy import sqrt,log
from random import choice, shuffle

# 建立蒙特卡洛树的节点，属性入下:
# 是否为己方     is_Self: bool
# 是否为先手     is_First: bool
# 是否为合并     mode:bool
# 当前局数       current_round: int
# 访问次数       ni: int
# 总和           Wi: float
# 父节点         parent: Node
# 可行的步       available: iter
# 已访问子节点   childs: list

# 特殊属性:
# operations: list     只有己方节点有此属性。用于在求结果时返回下一步操作。
# strs :dict           只有敌方节点有此属性。用于在对方走完后继续搜索。


class Node:
    'MCTS树的节点'

    def __init__(self, is_Self:bool, mode:bool, current_Round:int, board, parent=None):
        '初始化'
        global Self_isFirst #己方是否为先手
        self.is_Self = is_Self #是否为己方
        self.is_First = not is_Self ^ Self_isFirst #当前节点是否为先手
        self.mode = mode #是否为合并
        self.board = board #棋盘
        self.current_Round = current_Round #当前局数
        self.parent = parent #父节点
        self.childs = [] #子节点
        self.ni = 0 #访问个数
        self.Wi = 0 #己方获胜次数

        if current_Round == 500:
            #到达底层
            self.available = iter([])
            if self.is_Self:
                self.operations = []
            else:
                self.strs = dict()
            return

        #得到全部可行的操作
        available = []
        if mode:
             #落子阶段
            self_pos = board.getNext(self.is_First, current_Round) #自己的落子点
            if self_pos:
                available.append(self_pos) #自己的
            available += board.getNone(not self.is_First) #对手的
        else:
           #合并阶段
            for direction in range(4): #四个方向
                if board.copy().move(self.is_First, direction):
                    available.append(direction)
        
        self.available = iter(available) #全部待生成节点的生成器
        if self.is_Self:
            self.operations = []
        else:
            self.strs = dict()


    def getNextChild(self) -> tuple:
        '返回UCB值最大的节点或没探索的节点'
        if self.available.__length_hint__():
            #情况1:有节点未被探索
            operation = next(self.available) #下一步操作
            #新子节点棋盘
            newboard = self.board.copy()
            if self.mode:
                newboard.add(self.is_First,operation)
            else:
                newboard.move(self.is_First,operation)
            #添加新子节点
            new_child = Node(
                not self.is_Self,
                not self.mode^self.is_First,
                self.current_Round + (not(self.is_First or self.mode)),
                newboard,
                self
            )
            self.childs.append(new_child)
            #在全部操作里加上新的操作
            if self.is_Self:
                self.operations.append(operation)
            else:
                self.strs[str(newboard)] = new_child
            
            return new_child, False #返回节点，停止循环
        

        if len(self.childs):
            #情况2: 为整棵树有叶节点且无未探索的节点
            return max(self.childs,key=UCT), True #返回UCB最大的节点，继续拓展
        

        if self.current_Round == 500:
            #情况3: 到最后一层了
            return self, False #返回自己，停止循环
            
        #情况4: 本回合无路可走
        new_child = Node(
            not self.is_Self,
            not self.mode^self.is_First,
            self.current_Round + (not(self.is_First or self.mode)),
            self.board.copy(),
            self
        )
        self.childs.append(new_child)
        return new_child, False #返回新节点，停止循环
    

    def getResult(self) -> tuple:
        "搜索结束，返回探索次数最多的节点"
        if len(self.operations):
            #普通情况:有操作时
            temporary_Root = self.childs[0] #新的根
            max_ni = temporary_Root.ni #最大的访问次数
            operation = self.operations[0] #操作
            index = 0 #索引

            for node in self.childs:
                if node.ni > max_ni:
                    max_ni = node.ni
                    operation = self.operations[index]
                    temporary_Root = node
                index += 1
            
            return operation, temporary_Root
        
        #当该节无法继续操作时
        return None, self.childs[0]
    

    def getNewRoot(self, board):
        "返回下一次探索的根节点"
        str_board = str(board)
        if str_board in self.strs:
            #情况1 已探索到
            node = self.strs[str_board]
            node.parent = None
            return node

        #情况2，该节点未探索
        return Node(
            not self.is_Self,
            not self.mode^self.is_First,
            self.current_Round + (not(self.is_First or self.mode)),
            board.copy()
        )




class Fakenode:
    "假节点，初始化时用"
    def getNewRoot(self, board):
        "初始化后第一次计算时用"
        return Node(True, True, 0, board.copy())



def UCT(node):
    "计算UCT"
    #己方情况，直接运算
    if node.is_Self:
        return node.Wi/node.ni + sqrt(2*log(Root.ni)/node.ni)
    
    #敌方情况，胜率变成负率
    return 1 - node.Wi/node.ni + sqrt(2*log(Root.ni)/node.ni)



def Quick(isFirst, mode, board, current_Round):
    "快速落子"
    if mode:  # 给出己方下棋的位置
        another = board.getNext(isFirst, current_Round)  # 己方的允许落子点
        if another != (): return another

        available = board.getNone(not isFirst)  # 对方的允许落子点
        if available:   # 整个棋盘已满
            board.add(isFirst, choice(available))
        return None

    else :  # 给出己方合并的方向
        directionList = [0, 1, 2, 3]
        shuffle(directionList)
        for direction in directionList:
            if board.move(isFirst, direction): 
                return None



def getWinner(board) -> bool:
    "判断胜利的一方"
    global Self_isFirst
    ours = board.getScore(Self_isFirst)
    theirs = board.getScore(not Self_isFirst)
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



class Player:
    "用蒙特卡洛树实现的player"

    def __init__(self, is_First:bool, array:tuple):
        "初始化"
        global Self_isFirst, Root
        Root = Fakenode()
        Self_isFirst = is_First
    
    def output(self, currentRound:int, board, mode:str):
        "返回下一步结果"
        global Self_isFirst, Root
        #根节点重置
        Root = Root.getNewRoot(board)
        for _ in range(50):
            #得到扩展的新节点
            Expansion_node = self.Selection_and_Expansion()
            #开始快速落子
            Win = self.Rollout(Expansion_node)
            #反向传播
            self.Backpropagation(Win, Expansion_node)
        #返回结果
        result, Root = Root.getResult()
        return result
    

    def Selection_and_Expansion(self) -> Node:
        "选择节点并拓展"
        global Root
        node = Root
        Continue = True
        while Continue:
            node, Continue = node.getNextChild()
        return node
    

    def Rollout(self, node:Node) -> bool:
        "通过快速落子模拟，返回胜负"
        current_Round = node.current_Round #局数
        mode = node.mode #模式
        isFirst = node.is_First #是否先手
        Board = node.board.copy() #棋盘
        #开始落子
        for _ in range(20):
            if current_Round == 500:
                break
            Quick(isFirst, mode, Board, current_Round)

            #下一轮状态
            isFirst ^= True
            mode ^= isFirst
            current_Round += isFirst and mode
        #判断胜负
        return getWinner(Board)


    def Backpropagation(self, Win:bool, node:Node):
        "反向传播"
        while node:
            node.ni += 1 #探索次数+1
            node.Wi += Win #己方获胜次数增加
            node = node.parent #继续处理父节点
