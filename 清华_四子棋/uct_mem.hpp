//
//  uct_mem.hpp
//  third_try
//
//  Created by 蔡 on 2020/4/15.
//  Copyright © 2020 蔡. All rights reserved.
//

#ifndef uct_mem_hpp
#define uct_mem_hpp

#include <iostream>
#include <vector>
#include <stack>

//在每一轮中更新棋局
inline void record(int x, int y, int type, int **_b, int *_t, int nX, int nY) {
    _b[x][y] = type;
    _t[y]--;
    if(y == nY and (x-1==nX)) {
        _t[y]--;
    }
}
inline void erase(int x, int y, int **_b, int *_t, int nX, int nY) {
    _b[x][y] = 0;
    _t[y]++;
    if(y == nY and (x-1==nX)) {
        _t[y]++;
    }
}

struct node_mem {   //！！这里是没存棋盘的，但是你的2048游戏的node节点可能需要存棋盘
    
    int cnt = 0;    //下的盘数
    int win = 0;    //本节点赢的盘数
    bool certain = false;   //是不是终盘的节点
    
    std::vector<node_mem* > children;   //子节点
    
    int depth;  //节点深度
    int next_line = -1; //扩展到哪一列了，如果值为N说明该节点已经扩展完成了
    int last_X; //上一步的落子处
    int last_Y;
    ~node_mem();
    void find_next_line(int *, int);
    
};

struct tree_mem {
    
    node_mem *root = nullptr;   //树根
    int **board = nullptr;      //棋盘
    int *top = nullptr;         //top数组，与棋盘统称棋局
    int nX, nY, M, N;
    
    std::stack<node_mem* >path; //记录每次扩展新节点时经过的路径，在回溯记分时使用
    int my_X = -1, my_Y = -1;   //记录自己上一次下起时的落子点，配合每一轮传进来的lastX、lastY，可以每一轮间更新棋局。
        
    
    ~tree_mem();
    bool new_game(const int *_b, int M, int N); //判断是不是新局面，靠数棋盘上棋子数
    void init(int _M, int _N, const int * _t, const int* __b, int _lX, int _lY, int _nX, int _nY);
    void tree_policy();
    void trace_back(int result);
    int default_policy();
    void choose(int &x, int &y);
};

#endif /* uct_mem_hpp */
