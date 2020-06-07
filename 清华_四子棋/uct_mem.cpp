//
//  uct_mem.cpp
//  third_try
//
//  Created by 蔡 on 2020/4/15.
//  Copyright © 2020 蔡. All rights reserved.
//

#include "uct_mem.hpp"
#include "Judge.h"
#include <random>
#include <cmath>
#include <iostream>
using namespace std;

node_mem::~node_mem() {
    for(auto e: children) {
        if(e) {
            delete e;
            e = nullptr;
        }
    }
}

void node_mem::find_next_line(int *top, int N) {
    do {
        next_line++;
    }while(next_line < N and top[next_line] <= 0);
}

bool tree_mem::new_game(const int *b, int _M, int _N) {
    int sum = 0;
    int end = _M-3;
    if(end < 0)
        end = 0;
    
    for(int i = _M-1; i >= end ; i--) {
        for(int j = 0; j < _N; j++) {
            if(b[i*_N+j]) {
                sum++;
            }
        }
        if(sum > 1)
            return false;
    }
    return true;
}

void tree_mem::init(int _M, int _N, const int *_t, const int *__b, int _lX, int _lY, int _nX, int _nY) {
    timespec emm;
    clock_gettime(CLOCK_REALTIME, &emm);
    
    srand(emm.tv_nsec);
    
    if(new_game(__b, _M, _N)){
        if(board) {
            for(int i = 0; i < M; i++) {
                delete []board[i];
            }
            delete []board;
            board = nullptr;
        }
        if(top) {
            delete []top;
            top = nullptr;
        }
        if(root) {
            delete root;
            root = nullptr;
        }
        my_X = my_Y = -1;
        M = _M; N = _N; nX = _nX; nY = _nY;
        board = new int* [M];
        for(int i = 0; i < M; i++) {
            board[i] = new int[N];
            for(int j = 0; j < N; j++) {
                board[i][j] = __b[i*N+j];
                if(board[i][j]) {
                    board[i][j] = 2;
                }
            }
        }
        top = new int [N];
        for(int i = 0; i < N; i++) {
            top[i] =_t[i];
        }
        root = new node_mem;
        root->depth = 0;
        root->find_next_line(top, N);
        
    }
    
    else {
        record(my_X, my_Y, 1, board, top, nX, nY);
        record(_lX, _lY, 2, board, top, nX, nY);
        node_mem* temp = nullptr;
        for(auto &e: root->children) {
            if(e->last_X == my_X and e->last_Y == my_Y) {
                for(auto &f: e->children) {
                    if(f->last_X == _lX and f->last_Y == _lY) {
                        temp = f;
                        f = nullptr;
                        break;
                    }
                }
                break;
            }
        }
        delete root;
        root = temp;
        if(!root) {
            root = new node_mem;
            root->depth = 0;
            root->find_next_line(top, N);
        }
    }
    
}

void tree_mem::tree_policy() {
    path.push(root);
    node_mem *bottom = root;
    while(bottom -> next_line == N) {
        
        node_mem *nxt = nullptr;
        double tot_max = 0;
        for(auto e: bottom->children) {
            
            double temp = 1-(e->win + 0.0)/e->cnt+sqrt(0.5*log(bottom->cnt)/e->cnt);
            
            if(tot_max < temp) {
                tot_max = temp;
                nxt = e;
            }
        }

        bottom = nxt;
        path.push(bottom);
        
        int type = 2;
        if(bottom -> depth%2) {
            type = 1;
        }
        
        record(bottom->last_X, bottom->last_Y, type, board, top, nX, nY);
        
    }
    if(!bottom -> certain) {
        
        int y = bottom->next_line;
        int x = top[y] - 1;
        int type = 1;
        if(bottom->depth % 2) {
            type = 2;
        }
        record(x, y, type, board, top, nX, nY);
        bottom->find_next_line(top, N);
        
        node_mem* pointer = new node_mem;
        pointer->depth = bottom->depth + 1;
        pointer->last_X = x;
        pointer->last_Y = y;
        pointer->find_next_line(top, N);
        bottom->children.push_back(pointer);
        path.push(pointer);
        
    }
}



int tree_mem::default_policy() {
    node_mem* node = path.top();
    int ans;
    
    
    int d = node->depth;
    int lx = node->last_X;
    int ly = node->last_Y;
    
    //如果局面上已经结束了，直接返回
    if(isTie(N, top)) {
        node->certain = true;
        return 0;
    }
    if(d%2) {
        if(userWin(lx, ly, M, N, board)) {
            node->certain = true;
            return 1;
        }
    }
    else {
        if(machineWin(lx, ly, M, N, board)) {
            node->certain = true;
            return 2;
        }

    }
    
    //创立临时棋盘
    int **_b = new int* [M];
    for(int i = 0; i < M; i++) {
        _b[i] = new int [N];
        for(int j = 0; j < N; j++) {
            _b[i][j] = board[i][j];
        }
    }
    int *_t = new int [N];
    for(int i = 0; i < N; i++) {
        _t[i] = top[i];
    }
    
    
    while(true) {
        
        int select = -1;
        bool flag = false;
        
        //寻找必胜，如果找到直接返回
        for(int i = 0; i < N; i++) {
            if(_t[i] <= 0) {
                continue;
            }
            int tempy = i;
            int tempx = _t[tempy]-1;
            int tempt = 1;
            if(d%2)
                tempt = 2;
            record(tempx, tempy, tempt, _b, _t, nX, nY);
            d++;
            if(d%2 and userWin(tempx, tempy, M, N, _b)) {
                select = tempy;
                erase(tempx, tempy, _b, _t, nX, nY);
                d--;
                ans = 1;
                flag = true;
                break;
            }
            if((d+1)%2 and machineWin(tempx, tempy, M, N, _b)) {
                select = tempy;
                erase(tempx, tempy, _b, _t, nX, nY);
                d--;
                ans = 2;
                flag = true;
                break;
            }
            
            d--;
            erase(tempx, tempy, _b, _t, nX, nY);
        }
        if(flag)
            break;
        
        //避免必败
        if(select == -1) {
            for(int i = 0; i < N; i++) {
                if(_t[i] <= 0) {
                    continue;
                }
                int tempy = i;
                int tempx = _t[tempy]-1;
                int tempt = 2;
                if(d%2)
                    tempt = 1;
                record(tempx, tempy, tempt, _b, _t, nX, nY);
                d++;
                if(d%2 and machineWin(tempx, tempy, M, N, _b)) {
                    select = tempy;
                    erase(tempx, tempy, _b, _t, nX, nY);
                    d--;
                    break;
                }
                if((d+1)%2 and userWin(tempx, tempy, M, N, _b)) {
                    select = tempy;
                    erase(tempx, tempy, _b, _t, nX, nY);
                    d--;
                    break;
                }
                d--;
                erase(tempx, tempy, _b, _t, nX, nY);
            }
        }
        
        //如果没有选择上，则随机落子
        if(select == -1) {
            while(true) {
                select = rand()%N;
                if(_t[select]>0)
                    break;
                
            }
        }
        
        
        int __x = _t[select]-1;
        int type = 1;
        if(d%2)
            type = 2;

        record(__x, select, type, _b, _t, nX, nY);
        d++;
        //只需要判断平局
        if(isTie(N, _t)) {
            ans = 0;
            break;
        }
        
    }
   
    
    for(int i = 0; i < M; i++) {
        delete []_b[i];
    }
    delete []_b;
    delete []_t;
    
    return ans;
    
};

void tree_mem::trace_back(int result) {
    while (!path.empty()) {
        node_mem* node = path.top();
        path.pop();
        if(node != root) {
            erase(node->last_X, node->last_Y, board, top, nX, nY);
            
        }
        node->cnt += 2;
        int d = node->depth;
        
        if(d%2) {
            if(result == 2) {
                node->win+=2;
            }
            else if(result == 0) {
                node->win+=1;
            }
        }
        else {
            if(result == 1) {
                node->win += 2;
            }
            else if(result == 0) {
                node->win+=1;
            }
        }
    }
}

void tree_mem::choose(int &x, int &y) {
    int max = 0;
    for(auto e: root->children) {
        if(max < e->cnt) {
            max = e->cnt;
            x = e->last_X;
            y = e->last_Y;
        }
    }
    my_X = x;
    my_Y = y;
}

tree_mem::~tree_mem() {
    delete root;
    root = nullptr;
    for(int i = 0; i < M; i++) {
        delete []board[i];
    }
    delete []board;
    delete []top;
    board = nullptr;
    top = nullptr;
    
}
