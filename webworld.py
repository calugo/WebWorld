#!/usr/bin/env python3
from webworld_lib import *


def main(args):

    #Maximum system size: Nsp
    Nsp = 100;
    #Traits Pool Size K  and Species Length
    K = 500; L = 10; 
    #Traits scores, Species Scores: M,S
    #Effort, Rates and Comp matrices: F,G,A 
    #Population Array: N
    #Species array: TR
    TR = -1*np.ones((Nsp,L),dtype=np.int32)
    M = Mab(K);
    G = np.zeros((Nsp, Nsp));  #Rates
    S = np.zeros((Nsp, Nsp));  #Scores
    A = np.zeros((Nsp, Nsp));  #Competition
    F = np.zeros((Nsp, Nsp));  #Foraging Efforts
    N = np.zeros(Nsp); #Populations
    ####################################################
    #INITIAL SETUP    
    #Model Parameters: Resources:R
    # Ecological Effciency: ld 
    # F.response: b
    # Competition parameter: c
    R = 1e5; ld = 0.1; 
    b = 5e-3; c = 0.5;
    #Initial Population: No
    #Minimum population allowed: Nd
    No = 1.0; Nd = 1.0;
    
    #Initial Setup:
    prm = np.array([K,L,R,ld,b]);
    so, s1, sio, fio = initial_setting(prm,M);

    N[0] = R/ld; N[1] = No;
    S[1,0] = sio; F[1,0] = fio; A[1,1] = 1.0;
    TR[0] = so; TR[1] =s1;


    #Ngen - number of evolutionary steps.
    dF = 1.0; DN = 1.0; Ngen = 50000; DT = 0.2;

    #Stationary value for n1 - used to test.
    n1s = ((ld*sio - b)*N[0])/sio
    ess = [100]
    ini = False
    for k in range(Ngen):
        print('generation: ',k)
        dl=0  
        
        if (k in ess) and (ini == False):
            save_matrices(N,F,S,TR,dl,'BLOBS')
            ini = True
            dl+=1

        while DN>1e-2:
            #fit = 0          
            while dF>0.1:
                F,G,dF = foraging_strategy(N,F,A,S,b)
            N, chk, sj, DN = pop_update(N,G,ld,DT,Nd)
            if chk == True:
                N,S,F,A,TR = pop_check(N,S,F,A,TR,sj)
            dF = 1.0
            if k in ess:
                save_matrices(N,F,S,TR,dl,'BLOBS')
                dl+=1
        DN=1.0
        if k in ess:
            save_matrices(N,F,S,TR,dl,'BLOBS')

        save_matrices(N,F,S,TR,k,'ARR')
        F,S,A,TR,N = speciation(M,TR,A,S,F,K,L,c,No,N);

        NA = nspecies(N)
        store_ns(NA,k)

        if NA == Nsp-1:
            break

    print("COMPLETED")

if __name__ == "__main__":
    main(sys.argv[1:])

