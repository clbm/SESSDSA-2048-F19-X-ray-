if __name__ == "__main__":
    from Source import Chessboard #棋盘
from Appraise import getmark,Min_mark,Max_Mark #(伪)评估函数

#Gamemode = 0: 先手落子
#Gamemode = 1: 后手落子
#Gamemode = 2: 先手合并
#Gamemode = 3: 后手合并
#self.mode = 0: 落子
#self.mode = 1: 合并

def simulation(depth:int, Round:int, Gamemode:int, is_max:bool, board, alpha = Min_mark, beta = Max_Mark, is_root = False):
    '模拟对战'
    ##################################
        #警告: 还未做防超时保险系统    
    ##################################
    max_depth = 1

    #大于最大深度时:
    if depth >= max_depth or Round == 500:
        alpha = beta = getmark(board)
        return (alpha, beta)


    #以下为一般情况
    mode = Gamemode // 2
    isFirst = 1^(Gamemode%2)



    if mode == 0:   #落子阶段
        #第一步，找出全部可行落子
        Self_part = board.getNext(isFirst,Round) #自己棋盘落子的位置
        #available为全部可行位置
        available = [Self_part] if Self_part else []
        available += board.getNone(not isFirst) #对手棋盘可落子的位置


        #第二步，迭代模拟

        #无合法移动的特殊情况
        if len(available) == 0:
            alpha, beta = simulation(
                depth = depth, #深度不变
                Round = Round, #局数不变
                Gamemode = Gamemode+1, #模式+1
                is_max = not is_max, #敌我翻转
                board = board, #棋盘不变
                alpha = alpha, #alpha保留
                beta = beta #beta保留
            )

            return alpha, beta


        #一般情况
        result = available[0]
        for position in available:
            #子树对决
            new_board = board.copy()
            new_board.add(isFirst,position)
            alpha_beta = simulation(
                depth = depth, #深度不变
                Round = Round, #局数不变
                Gamemode = Gamemode+1, #模式+1
                is_max = not is_max, #敌我翻转
                board = new_board, #棋盘更新
                alpha = alpha, #alpha保留
                beta = beta #beta保留
                )

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

            #子树对决
            No_available = False
            alpha_beta = simulation(
                depth = depth + (1^isFirst),
                Round = Round + (1^isFirst),  #局数后手+1
                Gamemode = (Gamemode+1)%4,  #模式+1后对4取余
                is_max = not is_max,  #敌我反转
                board = new_board, #新的棋盘
                alpha = alpha, #alpha保留
                beta = beta #beta保留
                )

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
            alpha, beta = simulation(
                depth = depth + (1^isFirst),
                Round = Round + (1^isFirst),  #局数后手+1
                Gamemode = (Gamemode+1)%4,  #模式+1后对4取余
                is_max = not is_max,  #敌我反转
                board = board, #棋盘不变
                alpha = alpha, #alpha保留
                beta = beta #beta保留
                )
            return alpha, beta

        else:
            #返回结果
            if is_root:
                return result
            else:
                return alpha, beta


if __name__ == "__main__":
    from random import randrange
    from time import time
    '''
    Board = Chessboard(tuple(randrange(720720) for i in range(500)))
    print(Board)
    for Gamemode in range(4):
        t_start = time()
        result = simulation(0,0,Gamemode,True,Board,is_root=True)
        print(f't{Gamemode+1} = {time()-t_start}')
        print(f'result = {result}')
        if Gamemode >= 2:
            Board.move(1^(Gamemode%2),result)
        else:
            Board.add(1^(Gamemode%2),result)
        print(Board)'''


    Board = Chessboard(tuple(randrange(720720) for i in range(500)))
    from Source import Chessman
    ARRAY = [
        [0,3,2,4,2,-4,-2,-11],
        [4,5,3,-2,-3,-1,-6,-3],
        [2,4,1,-7,-6,8,-1,-2],
        [1,3,5,-2,-3,-5,-4,0]
    ]
    for i in range(4):
        for j in range(8):
            value = ARRAY[i][j]
            if value!=0:
                Board.board[(i,j)] = Chessman(value>0,(i,j),abs(value))
    print(Board)

    print(simulation(0,193,0,True,Board,is_root=True))