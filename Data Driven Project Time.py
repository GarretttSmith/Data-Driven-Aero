

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def readfile(file, alpha, area):
    f1 = open(file,'r')
    lines = f1.readlines()
    f1.close
    df = pd.read_csv(file,sep='\t',header=(0))
    m = df.to_numpy()
    #print(len(m))

    k = 0
    atm = 2116
    x = np.zeros(len(m))
    y = np.zeros(len(m))
    z = np.zeros(len(m))
    t = []
    Vel = np.zeros((len(m),5))

    for line in lines:

                line = line.strip().replace("'", "").replace('"', '').replace("(", "").replace(")", "")

                cells = line.split('\t')


                for i in range(len(cells)):
                    cells[i] = cells[i].strip()



                if k >= 1:
                    
                    Vel[k-1][0] = (float(cells[1]))
                    Vel[k-1][1] = (float(cells[2]))
                    Vel[k-1][2] = (float(cells[3]))
                    Vel[k-1][3] = (float(cells[4]))
                    Vel[k-1][4] = (float(cells[5]))
                    t.append(float(cells[0]))

                k +=1

    
    x1 = Vel[:,:-1]
    x2 = Vel[:,1:]
    dt = t[1] - t[0]
    #print(Vel)


    return x1, x2, Vel, dt, t

def SVD(Vel, x1, x2, dt, t):

    
     
    U, Sigma, VT = np.linalg.svd(x1,full_matrices=False)
    #print(U.shape,Sigma.shape, VT.shape)

    
    #plt.figure(figsize=(4, 4))
    #plt.plot(Sigma[:5], 'o-')
    #plt.xlabel('Index')
    #plt.ylabel('Singular Value')
    #plt.title('First 10 Singular Values of X')
    #plt.show()

    U, Sigma, VT = U[:, :2], Sigma[:2], VT[:2, :]


    A_tilde = np.linalg.multi_dot([U.conj().T, x2, VT.conj().T, np.linalg.inv(np.diag(Sigma))])
    Lambda, W = np.linalg.eig(A_tilde)
    L = np.diag(Lambda)
    #print(W.shape, L.shape)

    Phi = np.linalg.multi_dot([x2, VT.conj().T, np.linalg.inv(np.diag(Sigma)), W])

    #print(Phi.shape)
    #print(x1.shape)
    #print(Sigma.shape,VT.shape)
    Alpha = np.diag(Sigma) @ VT[:,0]
    #print(Alpha.shape)
    b = np.linalg.lstsq(Phi, x1[:,0], rcond=None)[0]
    b2 = np.linalg.solve(W @ L, Alpha)
    #print(b, b2)

    Omega = np.log(abs(Lambda))/dt

    t_exp = np.arange(Vel.T.shape[1]) * dt
    #print(t_exp.shape)
    #t_total = t_exp[146]
    temp = np.repeat(Omega.reshape(-1,1), t_exp.size, axis=1)

    #print(b.shape,temp.shape)
    dynamics = np.exp(temp * t_exp) * b.reshape(b.shape[0], -1)
    #print(dynamics.shape)

    X_dmd = Phi @ dynamics
    X_dmd = X_dmd[:,:5]
    #print(X_dmd)
    #print(Vel.shape)

    

    y = np.linspace(min(Vel[:,0]), max(Vel[:,0]), len(Vel[0,:]))

    plt.figure(figsize=(8, 4))

    #plt.subplot(1, 2, 1)
    #plt.contourf(t, y, np.real(Vel.T), 20, cmap='RdGy')
    #plt.colorbar()
    #plt.xlabel('time')
    #plt.ylabel('Velocity')
    #plt.title('Contour plot of X')
#
    #plt.subplot(1, 2, 2)
    #plt.contourf(t, y, np.real(X_dmd.T), 20, cmap='RdGy')
    #plt.colorbar()
    #plt.xlabel('Time')
    #plt.ylabel('Velocity')
    #plt.title('Contour plot of X_dmd')

    plt.plot(t, Vel[:,0], label = 'Actual Values')
    plt.plot(t, X_dmd[:,0], label = 'Approximated Values')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Acurracy Check')

    plt.show()

    #print(Vel - X_dmd)
    

    return

def main():

    File1 = 'Velocity Field Full.txt'
    

    Alpha1 = 0
    
    Wing_Area = 3
     
    x1, x2, Vel, dt, t = readfile(File1, Alpha1, Wing_Area)
    
    SVD(Vel, x1, x2, dt, t)

main()

