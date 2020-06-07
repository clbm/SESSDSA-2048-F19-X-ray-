import sys
import math
import random
import numpy as np
def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
def getWinner(board):#得到胜者
    pass
def checkMove(move):#步骤是否合法
    pass
class TreeNode(object):
    def __init__(self, parent, P):
        self.parent = parent
        self.children = {}
        self.visits = 0
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
    def expand(self,action_P):
        '''
        生成子节点
        :param action_P: 元祖（操作，先验概率） action=[position,direction]
        :return:
        '''
        for action,p in action_P:
            if action not in self.children:
                self.children[action]=TreeNode(self,p)
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
    def __init__(self,policy,c_puct,playout=1000):
        '''

        :param policy: 策略函数 输入 期盘 输出（操作（棋子，移动），先验函数）
        :param c_puct: 常数
        :param playout: 执行次数
        '''
        self.root=TreeNode(None,1.0)
        self.policy=policy
        self.c_puct=c_puct
        self.playout=playout
    def _playout(self,board,isFirst):
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
            board.add(action[0])#下棋
            board.move(action[1])#移动
            action_P, leafval= self.policy(board)
            end=board.getNext()==() and board.getNone()==()
            if not end:
                node.expand(action_P)
            else:#结束
                winner = getWinner(board)
                if winner ==None: #平局
                    leafval = 0.0
                else:
                    leafval = 1.0 if winner==isFirst else -1.0 #判断输赢
            node.updateRecursive(-leafval)#更新祖先


    def getMoveAndProbs(self, board,isFirst, temp=1e-3):
        '''
        获得所有可行动作及其先验概率
        :param board:棋盘
        :param temp:常数
        :return:
        '''
        for n in range(self.playout):
            boardcopy=board.copy()
            self._playout(board,isFirst)
        #所有动作及概率
        actionAndvisits=[(action,node.visits)
                         for action, node in self.root.children.items()]
        actions,visits=zip(*actionAndvisits)
        P=softmax(1.0/temp*np.log(np.array(visits)+1e-10))
        return actions,P
    def updataMTCS(self,move):
        '''
        更新子树
        :param move:
        :return:
        '''
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)











