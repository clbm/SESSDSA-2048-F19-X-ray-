Min_mark = -1<<20
Max_Mark = 1<<20
Dict = {1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 64, 7: 128, 8: 256, 9: 512, 10: 1024, 11: 2048}
def getmark(board,isFirst):
    '评估函数'
    #part1:棋子直接估值部分
    #获得两方棋子数值的从大到小的列表
    ours = board.getScore(isFirst)
    theirs = board.getScore(not isFirst)
    zeros = -len(ours)+len(theirs)
    result = zeros
    for chess in ours:
        result += Dict[chess]
    for chess in theirs:
        result -= Dict[chess]
    return result
    

def simulation(depth:int, Round:int, Gamemode:int, is_max:bool, board, alpha = Min_mark, beta = Max_Mark, is_root = False):
    '模拟对战'
    max_depth = 6
    mode = Gamemode // 2
    isFirst = 1^(Gamemode%2)
    
    #大于最大深度时:
    if depth >= max_depth or Round == 500:
        alpha = beta = getmark(board,isFirst)
        return (alpha, beta)

    #以下为一般情况
    if mode == 0:   #落子阶段
        #第一步，找出全部可行落子
        Self_part = board.getNext(isFirst,Round) #自己棋盘落子的位置
        #available为全部可行位置
        available = [Self_part] if Self_part else []
        available += board.getNone(not isFirst) #对手棋盘可落子的位置

        #第二步，迭代模拟

        #无合法移动的特殊情况
        if len(available) == 0:
            alpha, beta = simulation(depth = depth,Round = Round,Gamemode = Gamemode+1,is_max = not is_max,board = board,alpha = alpha,beta = beta)
            #深度不变 #局数不变 #模式+1 #敌我翻转 #棋盘不变 #alpha保留 #beta保留
            return alpha, beta


        #一般情况
        result = available[0]
        for position in available:
            #子树对决
            new_board = board.copy()
            new_board.add(isFirst,position)
            alpha_beta = simulation(depth = depth+1,Round = Round,Gamemode = Gamemode+1,is_max = not is_max,board = new_board,alpha = alpha,beta = beta)
            #深度不变 #局数不变 #模式+1 #敌我翻转 #棋盘更新 #alpha保留 #beta保留

            #更新alpha-beta
            if is_max:
                if is_root:
                    #对于树根，需要比较alpha值的变化
                    old_alpha = alpha
                alpha = max(alpha, *alpha_beta)
                if is_root and alpha > old_alpha:
                    #当alpha值变大时，落子位置更新
                    result = position
            else:
                beta = min(beta, *alpha_beta)
                
            #alpha-beta剪枝
            if alpha >= beta:
                break
        
        #返回结果
        if is_root:
            return result
        else:
            return alpha, beta

    else:   #合并阶段
        No_available = True
        for move in range(4):
            #跳过非法合并
            new_board = board.copy()
            if not new_board.move(isFirst, move):
                continue
            elif No_available:
                No_available = False
                if is_root:
                    result = move

            #子树对决
            alpha_beta = simulation(depth = depth + 1,Round = Round + (1^isFirst),Gamemode = (Gamemode+1)%4,is_max = not is_max,board = new_board,alpha = alpha,beta = beta)  
                 #局数后手+1 #模式+1后对4取余 #敌我反转 #新的棋盘 #alpha保留 #beta保留

            #更新alpha-beta
            if is_max:
                if is_root:
                    #对于树根，需要比较alpha值的变化
                    old_alpha = alpha
                alpha = max(alpha, *alpha_beta)
                if is_root and alpha > old_alpha:
                    #当alpha值变大时，落子位置更新
                    result = move
            else:
                beta = min(beta, *alpha_beta)
                
            #alpha-beta剪枝
            if alpha >= beta:
                break               
            
        #无合法移动的特殊情况
        if No_available:
            alpha, beta = simulation(depth = depth + 1,Round = Round + (1^isFirst),Gamemode = (Gamemode+1)%4,is_max = not is_max,board = board,alpha = alpha,beta = beta)  
            #局数后手+1 #模式+1后对4取余 #敌我反转 #棋盘不变 #alpha保留 #beta保留
            return alpha, beta

        else:
            #返回结果
            if is_root:
                return result
            else:
                return alpha, beta


class Player:
    def __init__(self, isFirst:bool, array:list):
        '''
        初始化
        参数: isFirst是否先手, 为bool变量, isFirst = True 表示先手
        参数: array随机序列, 为一个长度等于总回合数的list'''
        self.isFirst = isFirst

    def output(self, currentRound:int, board, mode:str):
        '''
        给出己方的决策(下棋的位置或合并的方向)
        参数: currentRound当前轮数, 为从0开始的int
        参数: board棋盘对象
        参数: mode模式, mode = 'position' 对应位置模式, mode = 'direction' 对应方向模式, 如果为 '_position' 和 '_direction' 表示在对应模式下己方无法给出合法输出
        返回: 位置模式返回tuple (row, column), row行, 从上到下为0到3的int; column列, 从左到右为0到7的int
        返回: 方向模式返回direction = 0, 1, 2, 3 对应 上, 下, 左, 右
        返回: 在己方无法给出合法输出时, 对返回值不作要求
        '''
        if mode == 'position':  # 给出己方下棋的位置
            Gamemode = 1^self.isFirst
        elif mode == 'direction':  # 给出己方合并的方向
            Gamemode = 2+(1^self.isFirst)
        else:
            return
        
        return simulation(depth = 0, Round = currentRound,Gamemode = Gamemode,is_max = True,board = board,is_root = True)
