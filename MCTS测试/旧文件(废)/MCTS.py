from Rollout import Rollout, getmark
from Structure import Node

def Selection(node:Node) -> Node:
    '探索加拓展'
    N = node.n #根节点的N
    explore = True
    while explore: #开始探索
        node, explore = node.Nextchild(N)
    return node


def Backpropagation(node:Node, dn:int, dq:float):
    '反向传播'
    backup = True
    while backup:
        node.n += dn #n变化
        node.q += dq #q变化
        node = node.parent
        backup = bool(node) #node为None时停止


def MCTS(root:Node, Gamemode:int, Round:int, explore_time:int = 100, rollertime = 20):
    '探索主函数'

    for _ in range(explore_time): #探索多次
        node = Selection(root) #选节点
        #到达底层，直接返回
        if node.Round == 499:
            dn = 1
            Q = getmark(node.board, 1-Gamemode%2)
        else:
            #进行模拟对局
            Q = 0
            dn = rollertime
            for _ in range(rollertime):
                Q += Rollout(Round, node.board.copy(), (node.Gamemode+1)%4, 1-root.Gamemode%2)
        Backpropagation(node, dn, Q)
    return root.Get_operation()


if __name__ == "__main__":
    from Source import Chessboard
    from random import randrange
    Board = Chessboard(tuple(randrange(720720) for i in range(50)))
    node = Node(0,0,Board)
    print(MCTS(node, 0, 0, 1))