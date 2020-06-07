#import sys
import math
import random
import numpy as np

def softmax(x):
    "返回概率"
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def getWinner(board,isFirst) -> bool:#得到胜者
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


def judgeDirection(isFirst, move1,direction,board) -> bool:
    '判断移动是否合法'
    boardcopy=board.copy()
    boardcopy.add(isFirst, move1)  # 下棋
    boardcopy.move(isFirst, direction)  # 移动
    if boardcopy==board:
        return False
    else:
        return True
    
def getLegalMoves(isFirst, moves1,directions,board):
    '获得合理的操作list'
    moves=[]
    for i in moves1:
        for j in directions:
            if judgeDirection(isFirst, i,j,board):
                moves.append([i,j])
            else:
                pass
    return moves
def policy(board, isFirst, currentRound):
    """
    :param board:
    :return:
    """
    # return uniform probabilities and 0 score for pure MCTS
    moves=getLegalMoves(isFirst, board.getNext(isFirst, currentRound),[0,1,2,3],board)
    action_probs = np.ones(len(moves))/len(moves)
    return zip(moves, action_probs), 0

class TreeNode(object):
    def __init__(self, parent,state ,P):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.state=state.copy()
        self.Q = 0
        self.u = 0
        self.P =P#先验概率
    def isRoot(self):
        return self.parent==None
    def isLeaf(self):
        return self.children=={}
    #选择
    def getValue(self,c_puct):
        self.u=(c_puct*self.P*np.sqrt(self.parent.visits)/(1+self.visits))
        return self.u+self.Q
    def select(self,c_puct):
        #选value最大的子节点
        return max(self.children.items(),
                   key=lambda node:node[1].getValue(c_puct))
    #扩展
    def expand(self, isFirst, action_P):
        '''
        生成子节点
        :param action_P: 元祖（操作，先验概率） action=[position,direction]
        :return:
        '''
        for action,p in action_P:
            if action not in self.children:
                board=self.state.copy()
                board.add(isFirst, action[0])  # 下棋
                board.move(isFirst, action[1])  # 移动
                self.children[action]=TreeNode(self,board,p)
    #回溯
    def updata(self,leafval):
        '''
        更新叶节点的属性
        :param leafval:
        :return:
        '''
        self.visits+=1
        self.Q=self.Q+1.0*(leafval-self.Q)/self.visits
    def updataRecursive(self,leafval):
        '''
        更新叶节点的所有祖先
        :param leafval:
        :return:
        '''
        current=self
        while not current.parent:
            current.parent.updata(-leafval)
            current=self.parent
class MCTS:
    def __init__(self,board,policy,c_puct,playout=1000):
        '''

        :param policy: 策略函数 输入 期盘 输出（操作（棋子，移动），得分）
        :param c_puct: 常数
        :param playout: 执行次数
        '''
        self.root=TreeNode(None,board,1.0)
        self.policy=policy
        self.c_puct=c_puct
        self.playout=playout
    def _playout(self,board,isFirst, currentRound):
        '''

        :param board: copy棋盘
        :return:
        '''
        node=self.root
        while True:
            if node.isLeaf():
                break
            #过程模拟 贪心算法
            action,node=node.select(self.c_puct)
            board.add(isFirst, action[0])#下棋
            board.move(isFirst, action[1])#移动
        action_P, leafval= self.policy(board)
        end=board.getNext(isFirst, currentRound)==() and board.getNone(not isFirst)==()
        if not end:
            node.expand(action_P)
        else:#结束
            winner = getWinner(board,isFirst)
            if winner ==None: #平局
                leafval = 0.0
            else:
                leafval = 1.0 if winner==isFirst else -1.0 #判断输赢
        node.updateRecursive(-leafval)#更新祖先


    def getMove(self, state,isFirst, currentRound):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        Return: the selected action
        """
        for _ in range(self.playout):
            statecopy = state.copy()
            self._playout(statecopy,isFirst, currentRound)
        return max(self.root.children.items(),
                   key=lambda node: node[1].visits)[0]


    def getMoveAndProbs(self, board,isFirst, currentRound ,temp=1e-3):
        '''
        获得所有可行动作及其先验概率
        :param board:棋盘
        :param temp:常数
        :return:
        '''
        for _ in range(self.playout):
            boardcopy=board.copy()
            self._playout(boardcopy,isFirst, currentRound)
        #所有动作及概率
        actionAndvisits=[(action,node.visits)
                         for action, node in self.root.children.items()]
        actions,visits=zip(*actionAndvisits)
        P=softmax(1.0/temp*np.log(np.array(visits)+1e-10))
        return actions,P
    def updataMTCS(self,lastmove,board):
        '''
        更新子树
        :param move:
        :return:
        '''
        if lastmove in self.root.children:
            self.root = self.root.children[lastmove]
            self.root.parent = None
        else:
            self.root = TreeNode(None,board,1.0)
class MCTSPlayer(object):
    "蒙特卡洛搜索树的接口(未完全)"
    def __init__(self,isFirst,board,policy=policy,c_puct=5, playout=2000,):
        self.mcts = MCTS(board,policy, c_puct, playout)
        self.isFirst= isFirst
    def setPlayer(self,p):
        self.player=p #没搞懂
    def reSetPlayer(self):
        pass #这个函数好像没啥用
    def getAction(self,board,isFirst, currentRound):#找最佳操作
        moves1=board.getNext(isFirst, currentRound)
        direction=[0,1,2,3]
        moves=getLegalMoves(isFirst, moves1,direction,board)
        if len(moves)>0:
            move=self.mcts.getMove(board,isFirst, currentRound)
            self.mcts.updataMTCS(move,board)
            return move
        else:
            return
    def output(self,currentRound, board, mode):
        if mode == 'position':  # 给出己方下棋的位置
            return self.getAction(board,self.isFirst, currentRound)[0]
        elif mode == 'direction':  # 给出己方合并的方向
            return self.getAction(board, self.isFirst, currentRound)[1]
        else:
            return


class Player:
    def __init__(self, isFirst:bool, array:tuple):
        self.isFirst = isFirst
        self.Inited = False
    
    def output(self, currentRound, board, mode):
        if self.Inited:
            pass
        else:
            self.player = MCTSPlayer(self.isFirst, board)
            self.Inited = True
        
        return self.player.output(currentRound, board, mode)
