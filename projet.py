import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint
import seaborn
from tqdm import tqdm


## Step 1 - Generate G1,...Gd n i.i.d. samples following the law of G

def n_corr_gaussians():
    G = np.random.normal(0,1,d)
    G = G.reshape((-1,1))

    return G

def corr_guassians():

    C = np.array([[int(i==j)+ rho*int(i!=j) for i in range(d)] for j in range(d)])
    L = np.linalg.cholesky(C)
    G = np.random.normal(0,1,d*N)
    G = G.reshape((-1,1))

    rep_L = np.tile(L,(1,N))
    W_TN = (sqrt(h) * rep_L.dot(G)).reshape((-1,1))
    somme = np.linalg.norm(L)
    W_TN = W_TN / somme * T

    return W_TN

def corr_Gn():
    Gn_T = np.zeros((d,n))

    for i in range(n):
        W_TN = corr_guassians()
        Gn_T[:,i] = W_TN[:,-1]

    return Gn_T

def n_corr_Gn():
    Gn_T = np.zeros((d,n))

    for i in range(n):
        W_TN = n_corr_gaussians()
        Gn_T[:,i] = W_TN[:,-1]

    return Gn_T


def BS(S0, W_TN):

    W_TN = W_TN.reshape((-1,1))
    S_TN = S0 * np.exp((r-(sigmas**2)/2) * T + sigmas * W_TN)
    S_TN = S_TN.reshape((-1,1))

    return S_TN

## Step 2 - Compute the minimizer

def f(G_T):

    d_G, nb_G = np.shape(G_T)
    S_TN = np.zeros((d,nb_G))
    for i in range(nb_G):
        S_TN[:,i] = BS(S0, G_T[:,i]).reshape(-1)
    maxi = np.amax(portofolio.dot(S_TN) - K, axis=0) #########################################################
    return np.where(maxi < 0, 0, maxi)

def grad_un_v (A, v, Gn_T):
    grad = (A.T).dot(A).dot(v).reshape((-1,1))
    temp = (f(Gn_T)**2).reshape((1,-1))
    somme = np.sum((A.T).dot(Gn_T) * temp * (np.exp(-(A.dot(v).reshape((-1,1))).T.dot(Gn_T)).reshape((1,-1))), axis=1)
    somme = somme / max(10,np.sum(temp * (np.exp(-(A.dot(v).reshape((-1,1))).T.dot(Gn_T)))))
    grad = grad - somme.reshape((-1,1))
    return grad

def hessian_un_v(A, v, Gn_T):
    hess = A.T.dot(A)
    temp = ((f(Gn_T)**2).reshape((1,-1)))
    somme1 = 0
    somme2 = 0
    somme3 = 0

    for i in range(n):
        Gi = Gn_T[:,i]
        Gi = Gi.reshape((-1,1))
        temp2 = (temp[0,i] * np.exp(-(A.dot(v).reshape((-1,1))).T.dot(Gi)))
        somme1 = somme1 + (A.T.dot(Gi).dot(Gi.T).dot(A)) * temp2
        somme2 = somme2 + temp2
        somme3 = somme3 + A.T.dot(Gi) * temp2
    somme2 = max(10e-6, somme2)
    hess = hess + somme1 /somme2 - somme3.dot(somme3.T) /((somme2)**2)
    return hess

# Newton's algorithm

def newton(A, x0, Gn_T, eps = 10**(-6)):
    xn_k = x0
    xn_k = xn_k.reshape((-1,1))
    k = 1
    g = grad_un_v(A, xn_k, Gn_T)

    nb_iter = 0
    while np.linalg.norm(g) > eps and nb_iter<15:
        h = hessian_un_v(A,xn_k,Gn_T)
        g = -grad_un_v(A,xn_k,Gn_T)
        d = np.linalg.inv(h).dot(g)
        #print("d\n", d)
        xn_k = xn_k + 1*d.reshape((-1,1))
        k += 1
        nb_iter += 1
        print("Norme du gradient :", np.linalg.norm(g))
    return xn_k

## Step 3 - Compute E[f(G)] by Monte Carlo

def expectation(Gn_T, A, x0, eps):

    xn = newton(A, x0, Gn_T, eps)
    xn = xn.reshape((-1,1))

    Mn = 0
    for i in range(n):
        Gi = Gn_T[:,i]
        Gi = Gi.reshape((-1,1))
        power = (A.dot(xn).T).dot(Gi/sqrt(T)) + np.linalg.norm(A.dot(xn))**2/2
        temp = f(Gi/sqrt(T) + A.dot(xn).reshape((-1,1))) * exp(-power)
        Mn += temp
    return exp(-r*T)/n*Mn

## Parameters :

d = 40              # actif
r = 0.05            # Facteur d'actualisation
T = 1               # maturite
S0 = 50             # Initialisation des modèles de BS
sigmas = np.array([[0.2] for i in range(d)])        # Sigma des modèles de BS
portofolio = np.array([[1/d for i in range(d)]])    # Répartition entre les différents BS
K = 45                                              # Strike
n = 5000                                           # Nombre de simulations pour Monte-Carlo
N = 100                                             # Nombre de points de discrétisation
rho = 0.1
h = sqrt(T/(N))


x0 = np.ones((d,1)).reshape((d,1))
A = np.identity(d)

print("Computing price with independant Brownian Motions... \n")
print("Computing of Gn ...")
G = n_corr_Gn()
print("Computing of E ...")
p1 = expectation(G,A,x0,10**(-6))
print("-> The price is {} \n".format(p1))

print("\n\n\n")
print("Computing price with correlated Brownian Motions... \n")
print("Computing of Gn ...")
Q = corr_Gn()
print("Computing of E ...")
p2 = expectation(Q,A,x0,10**(-6))
print("-> The price is {} \n".format(p2))


## Courbe d'influence de r :

print("\nInfluence de r :\n")

plt.figure()
r_s = np.linspace(0.03,0.1,20)
p_n_c = []
p_c = []
for i in tqdm(range(len(r_s))) :
    r = r_s[i]
    p1 = expectation(G,A,x0,10**(-6))
    print(" ")
    p2 = expectation(Q,A,x0,10**(-6))
    p_n_c.append(p1)
    p_c.append(p2)
plt.plot(r_s, p_n_c, label="Actifs non corrélés", )
plt.plot(r_s, p_c, label="Actifs corrélés")
plt.xlabel("r : taux d'actualisation", fontsize=12)
plt.ylabel("p : prix de l'option sur indice", fontsize=12)
plt.legend(fontsize=12)

## Courbe d'influence de K :

print("\nInfluence du Strike K :\n")

plt.figure()
r = 0.05
K_s = np.linspace(0.8*S0, 1.2*S0,20)
p_n_c = []
p_c = []
for i in tqdm(range(len(K_s))) :
    K = K_s[i]
    p1 = expectation(G,A,x0,10**(-6))
    print(" ")
    p2 = expectation(Q,A,x0,10**(-6))
    p_n_c.append(p1)
    p_c.append(p2)
plt.plot(K_s, p_n_c, label="Actifs non corrélés", )
plt.plot(K_s, p_c, label="Actifs corrélés")
plt.xlabel("K : Strike", fontsize=12)
plt.ylabel("p : prix de l'option sur indice", fontsize=12)
plt.legend(fontsize=12)


## Courbe d'influence de sigams :

print("\nInfluence de sigma :\n")

plt.figure()
r = 0.05
K = 45
sigma_s = np.linspace(0, 1,20)
p_n_c = []
p_c = []
for i in tqdm(range(len(K_s))) :
    sigma = sigma_s[i]
    sigmas = np.array([[sigma] for i in range(d)])
    p1 = expectation(G,A,x0,10**(-6))
    print(" ")
    p2 = expectation(Q,A,x0,10**(-6))
    p_n_c.append(p1)
    p_c.append(p2)
plt.plot(sigma_s, p_n_c, label="Actifs non corrélés", )
plt.plot(sigma_s, p_c, label="Actifs corrélés")
plt.xlabel("sigma ", fontsize=12)
plt.ylabel("p : prix de l'option sur indice", fontsize=12)
plt.legend(fontsize=12)


## Courbe d'influence de la dimension :

print("\nInfluence du nombre d'actifs :\n")

plt.figure()
r = 0.05
K = 45
sigmas = np.array([[0.2] for i in range(d)])
d_s =  np.array(np.linspace(1,100, 20), dtype=int)
p_n_c = []
p_c = []
for i in tqdm(range(len(d_s))) :
    d = d_s[i]
    A = np.identity(d)
    x0 = np.ones((d,1)).reshape((d,1))
    sigmas = np.array([[0.2] for i in range(d)])        # Sigma des modèles de BS
    portofolio = np.array([[1/d for i in range(d)]])
    G = n_corr_Gn()
    Q = corr_Gn()
    p1 = expectation(G,A,x0,10**(-6))
    print(" ")
    p2 = expectation(Q,A,x0,10**(-6))
    p_n_c.append(p1)
    p_c.append(p2)
plt.plot(d_s, p_n_c, label="Actifs non corrélés", )
plt.plot(d_s, p_c, label="Actifs corrélés")
plt.xlabel("d : nombre d'actifs considérés ", fontsize=12)
plt.ylabel("p : prix de l'option sur indice", fontsize=12)
plt.legend(fontsize=12)

## Courbe d'influence de N:

print("\nInfluence du nombre de N :\n")

plt.figure()
r = 0.05
K = 45
sigmas = np.array([[0.2] for i in range(d)])
d = 40
A = np.identity(d)
x0 = np.ones((d,1)).reshape((d,1))
sigmas = np.array([[0.2] for i in range(d)])        # Sigma des modèles de BS
portofolio = np.array([[1/d for i in range(d)]])
N_s = np.array(np.linspace(1,1000, 20), dtype=int)
p_n_c = []
p_c = []
for i in tqdm(range(len(N_s))) :
    N = N_s[i]
    print("N",N)
    h = sqrt(T/(N))
    G = n_corr_Gn()
    Q = corr_Gn()
    p1 = expectation(G,A,x0,10**(-6))
    print(" ")
    p2 = expectation(Q,A,x0,10**(-6))
    p_n_c.append(p1)
    p_c.append(p2)
plt.plot(N_s, p_n_c, label="Actifs non corrélés", )
plt.plot(N_s, p_c, label="Actifs corrélés")
plt.xlabel("N : Finesse de la discrétisation en temps", fontsize=12)
plt.ylabel("p : prix de l'option sur indice", fontsize=12)
plt.legend(fontsize=12)

## Influence de rho :

print("\nInfluence du nombre de rho :\n")

plt.figure()
r = 0.05
K = 45
sigmas = np.array([[0.2] for i in range(d)])
d = 40
A = np.identity(d)
x0 = np.ones((d,1)).reshape((d,1))
sigmas = np.array([[0.2] for i in range(d)])        # Sigma des modèles de BS
portofolio = np.array([[1/d for i in range(d)]])
N = 10
h = sqrt(T/(N))
rho_s = np.linspace(-1/(d-1), 0.9, 20)
p_n_c = []
p_c = []
for i in tqdm(range(len(N_s))) :
    rho = rho_s[i]
    G = n_corr_Gn()
    Q = corr_Gn()
    p1 = expectation(G,A,x0,10**(-6))
    print(" ")
    p2 = expectation(Q,A,x0,10**(-6))
    p_n_c.append(p1)
    p_c.append(p2)
plt.plot(N_s, p_n_c, label="Actifs non corrélés", )
plt.plot(N_s, p_c, label="Actifs corrélés")
plt.xlabel("rho : facteur de corrélation entre les variables ", fontsize=12)
plt.ylabel("p : prix de l'option sur indice", fontsize=12)
plt.legend(fontsize=12)

plt.show()

## Comparaison Variance sans fonction d'imortance et avec :




## - Understanding the problem

def simulation():
    S = 0
    for k in range(n):
        x = S0*exp(sigmas[0]*sqrt(T)*np.random.normal(0,1)+(r-sigmas[0]**2/2)*T)
        if x > K:
            S+=1
    return S/n

def gaussian(x):
    return 1/(sqrt(2*np.pi))*exp(-x**2/2)

def probabilite(y):
    return 1 - quad(gaussian,-5000,y)[0]

y = 1/(sqrt(T)*sigmas[0])*(log(K/S0)-(r-sigmas[0]**2/2)*T)

plt.xlabel("Différence entre S0 et K")
plt.ylabel("Probabilité d'être dans la monnaie")
p = []
for k in range(S0,S0*2):
    p.append(probabilite(1/(sqrt(T)*sigmas[0])*(log(k/S0)-(r-sigmas[0]**2/2)*T)))
plt.plot(p)
#plt.show()
