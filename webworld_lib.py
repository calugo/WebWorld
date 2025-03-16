try:
    import numpy as np
    import os, sys
    import pandas as pd
    import numpy as np
    from numba import jit

except ImportError as e:
    sys.stderr.write("Error loading module: %s\n"%str(e))
    sys.exit()


@jit(nopython = True)
def gaussianC(C):

    if C != 1.0:
        Cj = np.random.uniform()
        if C<Cj:
            y = np.random.normal()
        else:
            y = 0.0
        return y
    
    else:
        
        return np.random.normal()
    return y

#@jit(nopython = True)
def Mab(K):

    A=np.ones((K,K))
    
    Iup = np.triu_indices(K);
    Idn = np.tril_indices(K);
    
    N = len(Iup[0])
    U = np.array([gaussianC(1.0) for j in range(N)])
    
    A[Iup] = U
    for i,j in zip(Idn[0],Idn[1]):
        A[i,j] = - A[j,i]
    
    np.fill_diagonal(A,0)
    return A


@jit(nopython = True)
def intersect(a,b):
    
    u = np.intersect1d(a,b);
    d = len(u)/len(a)
    return d

@jit(nopython = True)
def comp_score(qij,c):
    return c+(1-c)*qij

@jit(nopython = True)
def Sij(a,b,M):
    
    Sab = 0.0
    for ia in a:
        S = M[ia,b]
        Sab+=np.sum(S)

    if Sab > 0.0:
        return Sab
    
    else:
        return 0.0


@jit(nopython = True)
def gen_string(L,K):
    
    u = K*np.ones(int(L))
    l = 0

    while l<L:
        nj = np.random.randint(0,int(K))
        if nj not in u:
            u[l]=nj
            l+=1
    return u.astype(np.int32)


@jit(nopython = True)
def initial_setting(prm,M):
    #prm = [K,L,R,ld,b,No];
    K = prm[0]; L = prm[1]; R = prm[2];
    ld = prm[3]; b = prm[4];
    S1o = 0.0;
    bo = b/ld;
    while (S1o <= bo):
        so =  gen_string(L,K);
        s1 =  gen_string(L,K);
        S1o = Sij(so,s1,M)/L;
        f1o = 1.0;   
    
    return so, s1, S1o, f1o

@jit(nopython = True)
def foraging_strategy(N,F,A,S,b):

    Fn = np.zeros(F.shape)
    G = np.zeros(F.shape)
    nk = np.where(N>0)[0]
    jn = np.max(nk)
    
    for i in nk:
        if(i>0):
            Sq = S[i,:]
            Fq = F[i,:]
            pk = np.where(Sq>0)[0]
            #print(i,Sq)
            #print(pk)
            for j in pk:
                Sk = np.where(S[:,j]>0)[0]
            #    print(Sk)
                sn = 0.0;
                for k in Sk:
                    #print(i,j,k)
                    sn+=A[k,i]*S[k,j]*F[k,j]*N[k]
                #print(sn)
                G[i,j]=Sq[j]*Fq[j]*N[j]/(b*N[j]+sn)
    
    for i in nk:
        if (i>0):
            SGij = np.sum(G[i,:])
            Fn[i,:] = G[i,:]/SGij
        jn = np.where(Fn[i,:]>0.0)
        for j in jn[0]:
            if Fn[i,j]<1e-6:
                Fn[i,j] =  1e-6
                    

    DF = np.max(np.abs((F-Fn).ravel()))

    return Fn, G, DF

@jit(nopython = True)
def pop_update(N,G,ld,DT,Nd):

    #print(N)
    rem = False;
    #sj = [];
    
    Np = np.zeros(N.shape)
    la = (1 - DT)*N;
    lb = ld*DT*np.sum(G,axis=1)*N
    V = np.zeros(N.shape)

    nk = np.where(N>0)[0]

    for j in nk:
        #print(j,G[:,j]*N)
        if j>0:
            V[j]=np.sum(G[:,j]*N)
    
    Np = la + lb - DT*V
    Np[0] = N[0]
    #print(V)
    #print(lb)
    #print(G)
    #print(np.sum(G,axis=0))
    #print(np.sum(G,axis=0)*N)

    U = Np[(Np!=0) & (Np<Nd)]
    if len(U)>0:
        sj =  np.where(((Np!=0) & (Np<Nd)))[0]
        rem = True
        #print(sj)
        Np[sj] = 0.0

    #print(Np)
    DN = np.max(np.abs(N-Np))
    #print(DN)

    return Np,rem, sj, DN

@jit(nopython = True)
def pop_check(N,S,F,A,TR,jn):
    #print('DELETION')
    #print(jn)
    #print(N)
    #input("")
    Ln = len(TR[0,:])
    for jd in jn:
    #jd = np.where((N<Nd) & (N!=0.0))
        N[jd] = 0.0
        S[jd,:] = 0.0
        S[:,jd] = 0.0
        F[jd,:] = 0.0
        F[:,jd] = 0.0
        A[jd,:] = 0.0
        A[:,jd] = 0.0
        TR[jd,:] = -1*np.ones(Ln,dtype=np.int32)

    #print("+++++")
    return N,S,F,A,TR  

@jit(nopython = True)
def speciation(M,TR,A,S,F,K,L,c,No,N):
    Tn = TR[:,0]

    nsj = np.min(np.where(Tn==-1)[0]) #Allocated new species 
    Na = np.where(Tn>-1)[0]
    Nb = Na[Na!=0] #excludes external species
    ns = np.random.choice(Nb) #parent species
    maxj = np.max(Nb) #index of largest non zero species

    #print(Nb) #species alive
    #print(ns,maxj) #parent and maxj
    #print(nsj) #allocation

    snew = False; NFT = False;
    new_traits = TR[ns,:].copy()

    #print('parent')
    #print(TR[ns])

    while(snew==False):
        while(NFT == False):
            nf = np.random.randint(0,K)
            if nf not in new_traits:
                jn = np.random.randint(0,L)
                new_traits[jn] = nf
                NFT = True
        n=0
        for q in Na:
            a = TR[q]
            dij = np.intersect1d(a,new_traits)
            lj = len(dij)
            if (lj == L):
                n+=1
        
        if n==0:
            snew = True
        else:
            NFT = False
    
    #print("NEW_SPECIES")
    #print(nf)
    #print(new_traits)

    

    ##Initialising scores and pop
    TR[nsj] = new_traits
    N[nsj] = No
    N[ns] -= 1 
    A[nsj][nsj] = 1.0

    ia = TR[nsj]
    
    for q in Na:
        ja = TR[q]

        A[nsj][q] = comp_score(intersect(ia,ja),c)
        A[q][nsj] = A[nsj][q]

        if q == 0:
       
            S[nsj][0] = Sij(ia,ja,M)
            if S[nsj][0] > 0.0:
               
                if F[ns][0] > 0.0:
                    F[nsj][0] = F[ns][0]
                else:
                    F[nsj][0] = np.random.uniform(1e-6,1)
        
        if q>0:
            S[nsj][q] = Sij(ia,ja,M)
            S[q][nsj] = Sij(ja,ia,M)

            if S[nsj][q] > 0.0:
                if F[ns][q] > 0.0:
                    F[nsj][q] = F[ns][q]
                else:
                    F[nsj][q] = np.random.uniform(1e-6,1)
            
            if S[q][nsj]>0.0:
                if F[q][ns] > 0.0:
                    F[q][nsj] = F[q][ns]
                else:
                    F[q][nsj] = np.random.uniform(1e-6,1)
    
    return F,S,A,TR,N

def report_system(TR,N,A,F,S):
    for q in range(len(TR[:,0])):
        print("=-=-=-=-=-=-=-")
        print(q,",",TR[q],",",N[q])
        print(S[q,:])
        print(F[q,:])
        print('-+-+-+-+-+-+')



def save_matrices(N,F,TR,S,nt,case):
    #case should be a folder in my case ARR or ESS depending
    #the scale I am storing


    if case == 'BLOBS':

        np.save(case+'/N'+str(nt)+'.npy',N)    
        np.save(case+'/F'+str(nt)+'.npy',F)
        #np.save(case+'/Sp'+str(nt)+'.npy',TR) 
        #np.save(case+'/S'+str(nt)+'.npy',S) 

    if case == 'ARR':

        np.save(case+'/N'+str(nt)+'.npy',N)    
        np.save(case+'/F'+str(nt)+'.npy',F)
        #np.save(case+'/Sp'+str(nt)+'.npy',TR) 
        #np.save(case+'/S'+str(nt)+'.npy',S) 
    
    if case == 'ESS':
        np.save(case+'/F'+str(nt)+'.npy',F)
        np.save(case+'/S'+str(nt)+'.npy',S) 
        np.save(case+'/N'+str(nt)+'.npy',N)    


@jit(nopython = True)
def nspecies(N):

    Na = np.where(N>0)[0]
    #ns =np.max(Na)
    NA= len(Na)
    
    return NA-1

def store_ns(ns,nt):
    with open('N.dat','a') as fn:
        fn.write(','.join((str(nt),str(ns)))+'\n' )

    