import numpy as np


#def Normal_Distro_Initializer(x,mean,std):
#    return np.random.normal(mean,std,size=len(x))





def Update_HMC(obj,mass,N_LeapFrog,LFSize):

    ## K(p) = p**2/(2*mom_std)
    ## U(y) = -y**2/(2)
    
    ## renew momentum wrt gaussian distro.
    std_p = 2*mass
    new_p = np.random.normal(0,std_p,size=obj.N)
    
    ## Couple with potiential (Ising model) to evolve y
    ## Leap Frog: 
    new_y = obj.y
    for l in range(N_LeapFrog):
        new_p = new_p - LFSize*0.5* (-new_y)
        new_y = new_y + LFSize*new_p/mass
        new_p = new_p - LFSize*0.5*(-new_y)

    ## M-H :
    dE = np.sum(new_p**2-obj.p**2)/std_p + obj.fx_GetU(new_y) - obj.fx_GetU(obj.y)
    print(new_p)
    #print(p_new)    
    A = np.exp(-dE/obj.T)
    if A >= 1:
        obj.y = new_y
        obj.x = np.sign(obj.y)
        obj.p = new_p
    elif np.random.uniform() < A:
        obj.y = new_y
        obj.x = np.sign(obj.y)
        obj.p = new_p


def Update_SSU(obj):
    ## rand loc <x,y>
    xy = np.random.randint(obj.L,size=2)

    ## Get neigh.idxs
    neigh = obj.where_allneigh(xy[0],xy[1]) ## y => 1,x => 0

    ## Get neigh confg
    h =  np.sum(obj.x[neigh])

    ## proposal :
    new_y = np.random.uniform(-1,1)

    ## calculate dE:
    st = xy[1]*obj.L + xy[0]
    dE =  (-np.sign(new_y) + obj.x[st])*h - (obj.y[st]**2 - new_y**2)/2

    ## Metropolis
    A = np.exp(-dE/obj.T) 
    if A >= 1:
        obj.y[st] = new_y
        obj.x[st] = np.sign(new_y)
    elif np.random.uniform() < A:
        obj.y[st] = new_y
        obj.x[st] = np.sign(new_y)




class Aug_Ising2D_CMC:
    def __init__(self,L,T):
        self.L = L
        self.N = L**2
        self.T = T
        ## augumented :
        self.y = np.ones(self.N)

        ## Ising config.:
        self.x = np.sign(self.y)

        ## momentum :
        self.p = np.zeros(self.N)
        
    
        ## This is for measurement:
        self.Typ = {'M':0,'M2':1,'M4':2,'E':3}
        self.Obs = np.zeros(len(self.Typ))
        self.MCS_cntr = 0


        #if self.is_hmc:
        #    self.p = np.random.normal(0,2*self.mom_std,size=self.N)
        
    def Reset(self):
        self.y = np.ones(self.N)
        self.x = np.sign(self.y)
        self.p = np.zeros(self.N)
        self.Clear_Measurement()
        

    def where_allneigh(self,x,y):
        return  np.array([y*self.L + (x + 1)%self.L,\
                          y*self.L + (x-1+self.L)%self.L,\
                          ((y+1)%self.L)*self.L + x,\
                          ((y-1+self.L)%self.L)*self.L + x])

    def allneigh_x(self,x,y):
        return self.x[self.where_allneigh(x,y)]
    
    def allneigh_y(self,x,y):
        return self.y[self.where_allneigh(x,y)]
        
    def fx_GetU_ising(self,y2):
        tmp = np.sign(y2).reshape((self.L,self.L))
        return -np.sum(tmp * (np.roll(tmp,1,axis=1) + np.roll(tmp,1,axis=0)))
    
    def fx_GetU(self,y2):
        Ag = np.dot(y2,y2)*0.5

        return Ag + self.fx_GetU_ising(y2)
    
    def fx_GetK(self,p2,mass):
        Ag = np.dot(p2,p2)/(2.*mass)
        return Ag    
    
    def fx_GetE(self,p2,mass,y2):
        return self.fx_GetK(p2,mass) + self.fx_GetU(y2)


    def Measurement(self,weight=1.):
        M2 = np.sum(self.x)/self.N
        M2 = M2**2

        self.Obs[self.Typ['M']]  += np.sqrt(M2)
        self.Obs[self.Typ['M2']] += M2
        self.Obs[self.Typ['M4']] += M2**2
        self.Obs[self.Typ['E']] += self.fx_GetU_ising(self.y) / self.N
        self.MCS_cntr += 1
        
    def Statistic(self):
        self.Obs /= self.MCS_cntr

    def Clear_Measurement(self):
        self.Obs *= 0
        self.MCS_cntr = 0

if __name__ == "__main__":

    L = 2
    T = 10.0
    EQUIN = 40000
    BINNUM = 10
    BINSZ  = 20000
    
    mass      = 2
    NLeapFrog = 10
    LeapStepSize = 0.04
    

    ##MC = Aug_Ising2D_CMC(L,T,is_hmc=1)
    MC = Aug_Ising2D_CMC(L,T)
    
    #print ( MC.GetE(MC.x) )
    BinData = []
    print ("Equi")
    for i in range(EQUIN):
        Update_HMC(MC,mass,NLeapFrog,LeapStepSize)
        #Update_SSU(MC)

    print ("statistic")
    for b in range(BINNUM):
        print ("Bin %d"%(b))
        for sz in range(BINSZ):
            Update_HMC(MC,mass,NLeapFrog,LeapStepSize)
            #Update_SSU(MC)
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
