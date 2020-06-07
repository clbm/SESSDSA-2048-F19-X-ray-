Inf = float('inf')
import numpy as np
Weight_L = np.array((1.0,1.05,1.1,1.15,1.2,1.2,1.2,1.2))
Weight_R = np.array((1.2,1.2,1.2,1.2,1.15,1.1,1.05,1.0))
def getmark(board, isFirst:bool):
    "评估函数"
    Value, Belong = np.array(board.getRaw()).transpose((2,0,1))
    return np.sum(((Belong<<Value)*Weight_L-((1-Belong)<<Value)*Weight_R)*(1+0.02*Value))*(1 if isFirst else -1)
def simulation(depth:int, Round:int, Gamemode:int, is_max:bool, board, alpha, beta): #模拟对战
    global Inf
    mode,isFirst = Gamemode>>1,not Gamemode&1
    if depth >= 6 or Round == 500: #大于最大深度时
        alpha = getmark(board,isFirst)
        return (alpha,alpha)
    #以下为一般情况
    if mode == 0:   #落子阶段
        #第一步，找出全部可行落子
        Self_part = board.getNext(isFirst,Round) #自己棋盘落子的位置
        #available为全部可行位置
        available = [Self_part] if Self_part else []
        if Round > 30:available += board.getNone(not isFirst) #对手棋盘可落子的位置
        #第二步，迭代模拟   
        if len(available) == 0: #无合法移动的特殊情况
            if not depth&1:return -Inf, -Inf
            return simulation(depth+1,Round,Gamemode+1,not is_max,board,alpha,beta)
        #一般情况
        result = available[0]
        for position in available: #子树对决
            new_board = board.copy()
            new_board.add(isFirst,position)
            alpha_beta = simulation(depth+1,Round,Gamemode+1,not is_max,new_board,alpha,beta)
            #更新alpha-beta
            if is_max:
                if not depth:old_alpha = alpha #对于树根，需要比较alpha值的变化
                alpha = max(alpha, *alpha_beta)
                if not depth and alpha > old_alpha: result = position #当alpha值变大时，落子位置更新
            else: beta = min(beta, *alpha_beta)
            if alpha >= beta: break#alpha-beta剪枝
        
        #返回结果
        if depth:return alpha, beta
        return result
    else:   #合并阶段
        No_available = True
        for move in range(4):
            #跳过非法合并
            new_board = board.copy()
            if not new_board.move(isFirst, move):continue
            elif No_available:
                No_available = False
                if not depth: result = move
            #子树对决
            alpha_beta = simulation(depth+1,Round+1-isFirst,(Gamemode+1)&3,not is_max,new_board,alpha,beta = beta)
            #更新alpha-beta
            if is_max:
                if not depth:old_alpha = alpha#对于树根，需要比较alpha值的变化 
                alpha = max(alpha, *alpha_beta)
                if not depth and alpha > old_alpha:result = move#当alpha值变大时，落子位置更新  
            else:beta = min(beta, *alpha_beta)
            if alpha >= beta:break#alpha-beta剪枝      
        #无合法移动的特殊情况
        if No_available:
            if not depth&1: return -Inf, -Inf
            return simulation(depth+1,Round + 1-isFirst,(Gamemode+1)&3,not is_max,board,alpha,beta)
        else:
            if depth: return alpha, beta
            return result
class Player:
    def __init__(self, isFirst:bool, array:list):
        self.isFirst = isFirst
    def output(self, currentRound:int, board, mode:str):
        if mode == 'position':Gamemode = 1^self.isFirst  # 给出己方下棋的位置  
        elif mode == 'direction':Gamemode = 2+(1^self.isFirst)  # 给出己方合并的方向  
        else:return
        return simulation(0, currentRound,Gamemode,True,board,-Inf,Inf)
