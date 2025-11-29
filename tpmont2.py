import numpy as np
import random

# --------------------------------------------------
# بيئة Labyrinth صغيرة وواضحة
# --------------------------------------------------
class Labyrinth:
    def __init__(self):
        self.size = 10
        self.grid = np.full((10,10), '0')
        self.grid[0,0] = 'S'
        self.grid[9,9] = 'G'
        self.grid[1,1] = '#'
        self.grid[1,2] = '#'
        self.grid[3,1] = 'P'

        self.actions = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
        self.reset()

    def reset(self):
        self.agent = [0,0]
        return tuple(self.agent)

    def is_valid(self,r,c):
        if r<0 or r>=10 or c<0 or c>=10: return False
        if self.grid[r,c]=='#': return False
        return True

    def step(self, a):
        dr,dc = self.actions[a]
        nr = self.agent[0]+dr
        nc = self.agent[1]+dc
        if self.is_valid(nr,nc):
            self.agent=[nr,nc]

        cell=self.grid[nr,nc]
        if cell=='G': return (nr,nc),10,True
        if cell=='P': return (nr,nc),-10,True
        return (nr,nc),-1,False

# --------------------------------------------------
# Monte-Carlo Control (First-Visit)
# --------------------------------------------------
def mc_control(env, episodes=2000, gamma=0.99, eps=0.1):
    Q = np.zeros((100,4))
    returns = {}

    for ep in range(episodes):
        s = env.reset()
        episode = []
        done=False

        # توليد الحلقة
        while not done:
            s_idx=s[0]*10+s[1]
            a = random.randrange(4) if random.random()<eps else np.argmax(Q[s_idx])
            s2,r,done = env.step(a)
            episode.append((s_idx,a,r))
            s = s2

        # حساب العوائد First-Visit
        G=0
        visited=set()
        for t in reversed(range(len(episode))):
            s_idx,a,r = episode[t]
            G = r + gamma*G
            if (s_idx,a) not in visited:
                visited.add((s_idx,a))
                returns.setdefault((s_idx,a),[]).append(G)
                Q[s_idx,a] = np.mean(returns[(s_idx,a)])

    return Q

# --------------------------------------------------
# تجربة الخوارزمية
# --------------------------------------------------
env = Labyrinth()
Q = mc_control(env, episodes=500)

# طباعة السياسة الناتجة
arrows = {0:'↑',1:'↓',2:'←',3:'→'}
policy = np.argmax(Q,axis=1)

for r in range(10):
    row=[]
    for c in range(10):
        if env.grid[r,c]=='#': row.append('#')
        elif env.grid[r,c]=='P': row.append('P')
        elif env.grid[r,c]=='G': row.append('G')
        elif env.grid[r,c]=='S': row.append('S')
        else:
            idx=r*10+c
            row.append(arrows[policy[idx]])
    print(" ".join(row))
