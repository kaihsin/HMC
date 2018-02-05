import numpy as np



class Aug_Ising2D_CMC:
    def __init__(self,L,T):
        self.L = L
        self.y = np.random.uniform(-1.,1.,size=self.L**2)
        self.x = np.sign(self.y)
        self.T = T

        self.Typ = {'M':0,'M2':1,'M4':2,'E':3}
        self.Obs = np.zeros(len(self.Typ))
        self.MCS_cntr = 0

    def reset(self):
        self.y = np.random.uniform(-1.,1.,size=self.L**2)
        self.x = np.sign(self.y)
        self.Clear_Measurement()

    def Evaluate(self,y2,weight=1.):
        E2 = GetE(y2)
        E  = GetE(self.y)

        A = np.exp(-(E2-E)/self.T) * weight
        if A >= 1 :
            return 1
        elif np.random.uniform() < A:
            return 1
        else :
            return 0

    def GetE(self,y2):
        """
            This is a private function
            Do not call directly.
        """
        Ag = -np.dot(y2,y2)*0.5

        tmp = np.sign(y2).reshape((self.L,self.L))
        E = -np.sum(tmp * (np.roll(tmp,1,axis=1) + np.roll(tmp,1,axis=0)))
        return Ag + E

    def Update(self,y2):
        self.y = np.copy(y2)


    def Algo_SSU(self,weight=1.):
        xy = np.random.randint(self.L,size=2)
        neigh = np.array([xy[0]*self.L + (xy[1] + 1)%self.L,\
                        xy[0]*self.L + (xy[1]-1+self.L)%self.L,\
                        ((xy[0]+1)%self.L)*self.L + xy[1],\
                        ((xy[0]-1+self.L)%self.L)*self.L + xy[1]])
        h =  np.sum(self.x[neigh])\

        st = xy[0]*self.L + xy[1]
        new_y = np.random.uniform(-1,1)

        dE =  (-np.sign(new_y) + self.x[st])*h + (self.y[st]**2 - new_y**2)/2

        A = np.exp(-dE/self.T) * weight
        if A >= 1:
            self.y[st] = new_y
            self.x[st] = np.sign(new_y)
        elif np.random.uniform() < A:
            self.y[st] = new_y
            self.x[st] = np.sign(new_y)


    def Measurement(self,weight=1.):
        M2 = np.sum(self.x)/self.L**2
        M2 = M2**2

        self.Obs[self.Typ['M']]  += np.sqrt(M2)
        self.Obs[self.Typ['M2']] += M2
        self.Obs[self.Typ['M4']] += M2**2
        self.Obs[self.Typ['E']] += self.GetE(self.x) / self.L**2
        self.MCS_cntr += 1
    def Statistic(self):
        self.Obs /= self.MCS_cntr

    def Clear_Measurement(self):
        self.Obs *= 0
        self.MCS_cntr = 0

if __name__ == "__main__":

    L = 8
    T = 10.0
    EQUIN = 20000
    BINNUM = 4
    BINSZ  = 20000

    MC = Aug_Ising2D_CMC(L,T)

    #print ( MC.GetE(MC.x) )
    BinData = []
    print ("Equi")
    for i in range(EQUIN):
        MC.Algo_SSU()

    print ("statistic")
    for b in range(BINNUM):
        print ("Bin %d"%(b))
        for sz in range(BINSZ):
            MC.Algo_SSU()
            MC.Measurement()
        MC.Statistic()
        # get Bin data:
        BinData.append(np.copy(MC.Obs))
        # clear
        MC.Clear_Measurement()

    BinData = np.array(BinData)
    print ("M    = %f , err = %f"%(np.mean(BinData[:,0]),np.std(BinData[:,0])))
    print ("M2   = %f , err = %f"%(np.mean(BinData[:,1]),np.std(BinData[:,1])))
    print ("M4   = %f , err = %f"%(np.mean(BinData[:,2]),np.std(BinData[:,2])))
    print ("E    = %f , err = %f"%(np.mean(BinData[:,3]),np.std(BinData[:,3])))
