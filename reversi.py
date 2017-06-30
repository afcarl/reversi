import numpy as np
import random
import copy

class Board():

    def __init__(self):
        self.board = np.array([[0 for i in range(8)] for j in range(8)])

    def reset(self):
        self.board = np.array([[0 for i in range(8)] for j in range(8)])
        self.board[3][3] = -1
        self.board[4][4] = -1
        self.board[3][4] = 1
        self.board[4][3] = 1
        return self.board.astype(np.float32)

    def turn(self, x, y, dx, dy, player):
        if x+dx<0 or 8<=x+dx or y+dy<0 or 8<=y+dy or self.board[y+dy][x+dx] != player*-1:
            return 0
        i = 2
        while(0<=x+dx*i and x+dx*i<8 and 0<=y+dy*i and y+dy*i<8):
            if self.board[y+dy*i][x+dx*i] == player:
                for j in range(i-1):
                    self.board[y+dy*(j+1)][x+dx*(j+1)] = player
                return i-1
            elif self.board[y+dy*i][x+dx*i] == player*-1:
                i += 1
            else:
                return 0
        return 0

    def check(self, x, y, dx, dy, player):
        if x+dx<0 or 8<=x+dx or y+dy<0 or 8<=y+dy or self.board[y+dy][x+dx] != player*-1:
            return False
        i = 2
        while(0<=x+dx*1 and x+dx*i<8 and 0<=y+dy*i and y+dy*i<8):
            if self.board[y+dy*i][x+dx*i] == player:
                return True
            elif self.board[y+dy*i][x+dx*i] == player*-1:
                i += 1
            else:
                return False
        return False

    def check_board(self,player):
        able = []
        for y in range(8):
            for x in range(8):
                if self.board[y][x] != 0:
                    continue
                try:
                    for dy in range(-1,2):
                        for dx in range(-1,2):
                            if dx == 0 and dy == 0:
                                continue
                            if self.check(x,y,dx,dy,player):
                                able.append(y*8+x)
                                raise Exception()
                except:
                    pass
        return able

    def step(self,action):
        x = action%8
        y = action/8
        self.board[y][x] = 1
        reward = 0
        for dy in range(-1,2):
            for dx in range(-1,2):
                if dx == 0 and dy == 0:
                    continue
                reward += self.turn(x,y,dx,dy,1)
        done = False
        b = self.board.reshape((1,64))
        b = b[0]
        if -1 not in b or 0 not in b or 1 not in b or len(self.check_board(-1))==0:
            done = True
            return self.board.astype(np.float32), reward, done
        #self.render()
        while(True):
            able = self.check_board(-1)
            try:
                w_action = random.choice(able)
            except:
                break
            x = w_action%8
            y = w_action/8
            self.board[y][x] = -1
            for dy in range(-1,2):
                for dx in range(-1,2):
                    if dx == 0 and dy == 0:
                        continue
                    reward -= self.turn(x,y,dx,dy,-1)
            if len(self.check_board(1)) != 0:
                break

        if -1 not in b or 0 not in b or 1 not in b or len(self.check_board(-1))==0 and len(self.check_board(1))==0:
            done = True
        #self.render()
        return self.board.astype(np.float32), float(reward), done

    def render(self):
        print(self.board)
