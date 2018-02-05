import numpy as np



class Ising2D_CMC:
    def __init__(self,L,T):
        self.L = L
        self.x = 2*np.random.randint(2,size=L**2) - 1
        self.T = T

        self.Typ = {'M':0,'M2':1,'M4':2,'E':3}
        self.Obs = np.zeros(len(self.Typ))
        self.MCS_cntr = 0

    def reset(self):
        self.x = 2*np.random.randint(2,size=L**2) - 1
        self.Clear_Measurement()

    def Evaluate(self,x2,weight=1.):
        E2 = GetE(x2)
        E  = GetE(self.x)

        A = np.exp(-(E2-E)/self.T) * weight
        if A >= 1 :
            return 1
        elif np.random.uniform() < A:
            return 1
        else :
            return 0

    def GetE(self,x2):
        """
            This is a private function
            Do not call directly.
        """
        tmp = x2.reshape((self.L,self.L))
        E = -np.sum(tmp * (np.roll(tmp,1,axis=1) + np.roll(tmp,1,axis=0)))
        return E

    def Update(self,x2):
        self.x = np.copy(x2)


    def Algo_SSU(self,weight=1.):
        xy = np.random.randint(self.L,size=2)
        h =  self.x[xy[0]*self.L + (xy[1] + 1)%self.L]\
           + self.x[xy[0]*self.L + (xy[1]-1+self.L)%self.L]\
           + self.x[((xy[0]+1)%self.L)*self.L + xy[1]]\
           + self.x[((xy[0]-1+self.L)%self.L)*self.L + xy[1]]
        st = xy[0]*self.L + xy[1]
        new_s = np.random.randint(2)

        dE =  (self.x[st] - new_s )*h


        A = np.exp(-dE/self.T) * weight
        if A >= 1:
            self.x[st] *= -1
        elif np.random.uniform() < A:
            self.x[st] *= -1

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

    MC = Ising2D_CMC(L,T)

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
