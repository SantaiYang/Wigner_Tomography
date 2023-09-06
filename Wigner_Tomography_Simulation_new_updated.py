import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from qutip import (basis, destroy, mesolve, sesolve, ptrace, qeye, tensor, wigner, states, displace, expect, coherent, fock, sigmax, sigmay,sigmaz, sigmam, Qobj, fidelity)
from numpy import linalg as LA

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.special import genlaguerre
from math import sqrt,factorial
from numpy.linalg import cond,svd
from scipy.optimize import fmin,check_grad,minimize
from IPython.display import display, clear_output
import time



class parityMapping:
    def __init__(self, wc, wa, g, N_cav, N_qb, gamma_phase, gamma_qubit, al=0.0):
        self.wc = wc
        self.wa = wa
        self.g = g
        self.N_cav = N_cav
        self.N_qb = N_qb
        self.gamma_phase = gamma_phase
        self.gamma_qubit = gamma_qubit
        self.al = al
        
    def a(self):
        """Returns the photon annilihation operator for the cavity"""
        return tensor(destroy(self.N_cav), qeye(self.N_qb))
    
    def sm(self):
        """Returns the photon annilihation operator for the qubit"""
        return tensor(qeye(self.N_cav), destroy(self.N_qb))
    
    def sx(self):
        '''sigmax operator of the qubit'''
        return tensor(qeye(self.N_cav), sigmax())
    
    def sy(self):
        '''sigmay operator of the qubit'''
        return tensor(qeye(self.N_cav), sigmay())
    
    def sz(self):
        '''sigmaz operator of the qubit'''
        return tensor(qeye(self.N_cav), sigmaz())
    
    def c_ops(self):
        """Returns the collapse operators"""
        return [tensor(qeye(self.N_cav), self.gamma_phase * sigmaz()), tensor(qeye(self.N_cav), self.gamma_qubit * sigmam())]
    
    def Hamiltonian(self, name):
        '''Choose the proper Hamiltonian'''
        if name == 'dispersive_rot':
            delta = np.abs(self.wa - self.wc)
            chi = self.g**2/delta
            return chi * self.sz() * self.a().dag() * self.a()
        if name == 'dispersive':
            delta = np.abs(self.wa - self.wc)
            chi = self.g**2/delta
            return self.wc * self.a().dag() * self.a() + chi * self.sz() * self.a().dag() * self.a() + 1/2 * self.wa * self.sz()
        if name == 'rot_two_level':
            delta = np.abs(self.wa - self.wc)
            chi = self.g**2/delta
            return (self.wc - self.wa) * self.a().dag() * self.a() + self.g * (self.a().dag() * self.sm() + self.a() * self.sm().dag())
        if name == 'two_level':
            return self.wc * self.a().dag() * self.a() + 1/2 * self.wa * self.sz() + self.g * (self.a().dag() + self.a()) * (self.sm() + self.sm().dag())
        if name == 'general':
            return (self.wc - self.wa) * self.a.dag() * self.a - self.al/2 * self.sm.dag() * self.sm.dag() * self.sm * self.sm + self.g * (self.a * self.sm.dag() + self.a.dag() * self.sm())
        else:
            raise Exception("Invalid name. Please choose from dispersive_rot, dispersive, rot_two_level, two_level, and general")
    
    def simulation(self, psi0, tlist, name):
        '''
        Return the quantum states at each time point
        psi0: Initial state of the cavity-qubit coupling system
        tlist: time points at which quantum states are recorded
        name: the name of the Hamiltonian that we choose
        '''
        H = self.Hamiltonian(name)
        result = mesolve(H, psi0, tlist, [self.c_ops()]).states
        return result
    
    def find_optimal_time(self, tlist, name, tol):
        '''
        This function finds the time when parity mapping occurs
        tlist: time evolution list
        name: the name of Hamiltonian
        tol: tolerance of the error
        '''
        H = self.Hamiltonian(name)
        N = self.N_cav
        M = self.N_qb
        psi0 = tensor(fock(N, 0), (basis(M, 0) + basis(M, 1))/np.sqrt(2))
        psi1 = tensor(fock(N, 1), (basis(M, 0) + basis(M, 1))/np.sqrt(2))
        output = mesolve(H, psi0, tlist)
        output1 = mesolve(H, psi1, tlist)
        for i in range(0, len(tlist)):
            qubit = ptrace(output.states[i], 1).overlap(ptrace(output1.states[i], 1))
            # if np.abs(qubit[0,0]) <= tol and np.abs(qubit[0,1]) <= tol and np.abs(qubit[1,0]) <= tol and np.abs(qubit[1,1]) <= tol:
            if qubit < tol:
                return tlist[i]
        return ('increase the number of ts in tlist')
    
    def find_optimal_theta(self, name, optimal_t, tol):
        '''
        This function returns the proper angle theta for sigma_theta which we need to take the expectation value of
        name: the name of Hamiltonian
        tol: tolerance of the error
        TODO: calling find_optimal_time() here could waste time, this function should probably take the optimal time as an input
        TODO: is this the optimal time always less than pi/2chi? does the find optimal time here always work?
        For dispersive_rot and dispersive H, can set the tolerance very small. Smaller the tolerance, more accurate the optimal time. All smaller than pi/2chi
        For two_level_rot, the smallest possible tolerance is 0.002, and the optimal time is slightly smaller than pi/2*chi (by 1ns)
        For two_level, the smallest possible tolerance is 0.002, and the optimal time is much smaller than pi/2chi (by 130ns)
        '''
        if name == 'dispersive_rot':
            tlist = np.linspace(0, np.pi/(2*chi), 3000) 
        else:
            tlist = np.linspace(0, optimal_t, 3000) 
        # Even fock state
        N = self.N_cav
        M = self.N_qb
        psi0 = tensor(fock(N, 2), (basis(M, 0) + basis(M, 1))/np.sqrt(2))
        # Odd fock state
        psi1 = tensor(fock(N, 1), (basis(M, 0) + basis(M, 1))/np.sqrt(2))
        H = self.Hamiltonian(name)
        output = mesolve(H, psi0, tlist)
        output1 = mesolve(H, psi1, tlist)

        # We will sweep from 0 to 2pi
        theta = np.linspace(0, 2 * np.pi, 5000)
        # Array for difference between expectation values of sigma_theta for even fock cavity state (e.g. |2>) and odd fock cavity state (e.g. |1>)
        diff = []
        # The below are for the non-two-level-approx H. 
        for i in range(0, len(theta)):
            # Rotation Matrix
            if M == 3:
                rot_m = Qobj([[np.exp(-1j * theta[i]), 0, 0], [0, np.exp(1j * theta[i]), 0], [0, 0, 1]])
                sx = Qobj([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
            if M == 2:
                rot_m = Qobj([[np.exp(-1j * theta[i]), 0], [0, np.exp(1j * theta[i])]])
                sx = sigmax()
            
            sigma_theta = rot_m * sx
            dif = expect(sigma_theta, ptrace(output.states[-1], 1)) - expect(sigma_theta, ptrace(output1.states[-1], 1))
            diff.append(dif)

        #fig, ax = plt.subplots(1,1)
        #ax.plot(theta, diff)
        angle = theta[np.argmax(diff)]
        #ax.scatter(angle, np.amax(diff), color = 'r')
        return angle
    
    def parity_mapping(self, psi0, alpha, name, angle, tlist, two_level_approx = True):
        '''This function returns the wigner value through parity mapping at alpha
        psi0: the initially prepared state without displacement
        alpha: displacement
        name: the Hamiltonian that will be used
        c_ops: Collapse Operator
        tol: tolerance that you can accept when finding the optimal parity mapping time
        two_level_approx: if we only consider two levels in the qubit. If not, we limit our discussion to three-level qubit
        TODO: tolerance appears to be unused here so it can be removed from the arguments
        '''
        N = self.N_cav
        M = self.N_qb
        d = tensor(displace(N, alpha), qeye(M))
        # if density_matrix == False: 
        #print(density_matrix)
        if psi0.type == 'ket':
            displaced_psi0 = d * psi0
        if psi0.type == 'oper':
            displaced_psi0 = d * psi0 * d.dag()
        #else:
        #    raise Exception('Invalid Input Quantum Object Type')
            
        if two_level_approx == True: 
            rot_m = Qobj([[np.exp(-1j * angle), 0], [0, np.exp(1j * angle)]])
            s_theta = rot_m * sigmax()
        else:
            rot_m = Qobj([[np.exp(-1j * angle), 0, 0], [0, np.exp(1j * angle), 0], [0, 0, 1]])
            s_theta = rot_m * Qobj([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    
        c_ops = self.c_ops()
        H = self.Hamiltonian(name)
        output = mesolve(H, displaced_psi0, tlist, c_ops, [])
        qubit_final = ptrace(output.states[-1], 1)
        wig = expect(s_theta, qubit_final) * 2/np.pi
        return wig
    
    def run(self, psi0, optimized_alphas, name, tol = 0.001, two_level_approx = True):
        '''This function returns the array of wigner values at optimized chosen alphas'''
        t_range = np.linspace(0, np.pi/(2*chi), 40000)
        optimal_t = self.find_optimal_time(t_range, name, tol = tol)
        angle = self.find_optimal_theta(name, optimal_t, tol = tol)
        chi = self.g**2/(np.abs(self.wa - self.wc))
        if name == 'dispersive_rot':
            tlist = np.linspace(0, np.pi/(2*chi), 3000)
        else:
            tlist = np.linspace(0, optimal_t, 3000)
        
        wig_vals = []
        for aid, alpha in enumerate(optimized_alphas):
            wig = self.parity_mapping(psi0, alpha, name, angle, tlist, two_level_approx = two_level_approx)
            wig_vals.append(wig)
            if aid % 10 == 0:
                print(aid)
        return np.array(wig_vals)
    


#FD = 7
#r = sqrt(FD)
#th = np.linspace(-np.pi, np.pi, 100)
best_cost = float('inf')


def wigner_mat_and_grad(disps, FD):
    """This function calculates the gradient matrix and wigner value matrix
    FD is the number of levels of the cavity we will consider when reconstructing the density matrix
    """
    ND = len(disps)
    wig_tens = np.zeros((ND,FD,FD),dtype=np.complex)
    grad_mat_r = np.zeros((ND,FD,FD),dtype=np.complex)
    grad_mat_i = np.zeros((ND,FD,FD),dtype=np.complex)
    
    B = 4*np.abs(disps)**2
    pf = (2/np.pi)*np.exp(-B/2.0)  # prefactor
    for m in range(FD):
        # calculate terms for n=m
        x = pf * np.real((-1)**m * genlaguerre(m, 0)(B))
        term_r = -4*disps.real*x
        term_i = -4*disps.imag*x
        if m > 0:
            y = 8*pf*(-1)**(m-1)*genlaguerre(m-1, 1)(B)  # first derivative of x
            term_r += disps.real * y
            term_i += disps.imag * y
        wig_tens[:, m, m] = x
        grad_mat_r[:, m, m] = term_r  # matrix is displacements * (laguerre + first derivative) real parts
        grad_mat_i[:, m, m] = term_i
        
        # calculate terms for n!=m (off-diagonal)
        for n in range(m+1, FD):
            pf_nm = sqrt(factorial(m) / float(factorial(n)))
            x = pf * pf_nm * (-1)**m * 2 * (2*disps)**(n-m-1) * genlaguerre (m, n-m)(B)
            term_r = ((n - m) - 4*disps.real*disps)*x
            term_i = (1j * (n - m) - 4*disps.imag*disps) * x
            if m > 0:
                y = 8 * pf * pf_nm * (-1)**(m-1)*(2*disps)**(n-m) *genlaguerre (m-1, n-m+1)(B)  # first derivative
                term_r += disps.real * y
                term_i += disps.imag * y

            wig_tens[:, m, n] = disps * x
            wig_tens[:, n, m] = (disps * x).conj()
            grad_mat_r[:, m, n] = term_r
            grad_mat_r[:, n, m] = term_r.conjugate()  # complex conjugate
            grad_mat_i[:, m, n] = term_i
            grad_mat_i[:, n, m] = term_i.conjugate()
    return (wig_tens.reshape((ND, FD**2)), grad_mat_r.reshape((ND, FD**2)), grad_mat_i.reshape((ND, FD**2)))

def cost_and_grad(r_disps, FD):
    N = len(r_disps)  # r_disps is all real parts of alphas, then all complex parts of alphas
    c_disps = r_disps[:N//2] + 1j*r_disps[N//2:]  # complex displacements
    M, dM_rs, dM_is = wigner_mat_and_grad(c_disps, FD)
    # print(dM_rs, dM_is)  # in general dM_rs and dM_is have complex number entries
    U, S, Vd = svd(M) # singular value decomposition M = U S Vd; U is unitary, S is diagonal, rows of Vd are eigenvectors
    NS = len(Vd)
    cn = S[0] / S[-1]  # condition number
    
    # Einstein summation convention acted on operands
    dS_r = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_rs, Vd.conj().T).real
    dS_i = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_is, Vd.conj().T).real
    
    # gradients of the condition number, split into real/imaginary parts
    grad_cn_r = (dS_r[0]*S[-1] - S[0]*dS_r[-1]) / (S[-1]**2)
    grad_cn_i = (dS_i[0]*S[-1] - S[0]*dS_i[-1]) / (S[-1]**2)
    
    return cn, np.concatenate((grad_cn_r,grad_cn_i))

def wrap_cost(disps, FD = 7):
    FD = FD
    global best_cost
    cost, grad = cost_and_grad(disps, FD)
    # print(grad)
    best_cost = min(cost, best_cost)
    #ax.clear()
    #ax.plot(disps[:n_disps],disps[n_disps:], 'k.')
    #ax.plot(r*np.cos(th), r*np.sin(th),'r--')

    #ax.set_title('Condition Number = %.1f' % (cost ,))
    clear_output(wait=True)
    #display(f)
    #print ’\r%s (%s)’ % (cost , best cost),
    return cost, grad

def optimized_alphas_simul(n_disps, FD):
    best_cost = float('inf')
    # Plotting for the results, returns cost and grad in a way the minimize function can handle
    init_disps = np.random.normal(0, 1, 2*n_disps) * 0.5
    init_disps[0] = 0
    init_disps[n_disps] = 0
    ret = minimize(wrap_cost, init_disps, method='L-BFGS-B', jac=True, options=dict(ftol=1e-6))
    print(ret.message)
    new_disps = ret.x[:n_disps] + 1j*ret.x[n_disps:]
    new_disps = np.concatenate(([0], new_disps))
    print(len(new_disps))
    return new_disps
    

def project_and_normalize_density_matrix(rho_uncons):
    """Take a density matrix that is possibly not positive semi-definite, and also not trace one, and 
    return the closest positive semi-definite density matrix with trace-1 using the algorithm in
    PhysRevLett.108.070502. Note this method assumes additive Gaussian noise
    """

    # make the density matrix trace one
    rho_uncons = rho_uncons / np.trace(rho_uncons)

    d = rho_uncons.shape[0]  # the dimension of the Hilbert space
    [eigvals_un, eigvecs_un] = np.linalg.eigh(rho_uncons)

    # If matrix is already trace one Positive Semi-Definite, we are done
    if np.min(eigvals_un) >= 0:
        # print 'Already PSD'
        return rho_uncons
    # Otherwise, continue finding closest trace one,
    # Positive Semi-Definite matrix through eigenvalue modification
    eigvals_un = list(eigvals_un)
    eigvals_un.reverse()
    eigvals_new = [0.0] * len(eigvals_un)
    i = d
    a = 0.0  # Accumulator
    while eigvals_un[i - 1] + a / float(i) < 0:
        a += eigvals_un[i - 1]
        i -= 1
    for j in range(i):
        eigvals_new[j] = eigvals_un[j] + a / float(i)
    eigvals_new.reverse()

    # Reconstruct the matrix
    rho_cons = np.dot(eigvecs_un, np.dot(np.diag(eigvals_new), np.conj(eigvecs_un.T)))

    return rho_cons

def reconstruct_density_matrix(name, psi0, new_disps, FD, pmap: parityMapping, tol = 0.001, two_level_approx = True, simulation = True, wig_vals = []):
    '''
    This function returns the reconstructed density matrix of the quantum state based on wigner values (after the convertion to a physical density matrix)
    name: name of the Hamiltonian
    psi0: Initial state
    new_disps: optimized displacements on the phase space
    FD: Number of levels in cavity that we choose to consider when constructing the density matrix
    tol: tolerance of error involved in finding optimal time & angle
    two_level_approx: if we only consider two levels in the qubit
    TODO: looks like two_level_approx argument is unused here
    '''
    Nph = FD
    if simulation:
        wig_vals = pmap.run(psi0, new_disps, name, tol = tol, two_level_approx = two_level_approx)
    else:
        wig_vals = wig_vals
    x_list = np.real(wig_vals)
    M = np.matrix(wigner_mat_and_grad(new_disps, FD)[0])
    pseudo = np.dot(LA.pinv(M), np.transpose(np.matrix(x_list)))
    pseudo = pseudo.reshape((Nph, Nph))
    pseudo = project_and_normalize_density_matrix(pseudo)
    pseudo = Qobj(pseudo)
    return pseudo

def gn(rho, n, N_cav, N_qubit):
    '''
    This function calculates the n-th order coherence of light
    rho: density matrix/state vector of system
    n: The order of coherence 
    TODO: since pmap is only used here to get the creation/annihilation operators I think you can probably just have an argument for N_cav and create
    the operators within this function. Otherwise a user has to create a whole parityMapping object just to use the a and a.dag()
    '''
    a = tensor(destroy(self.N_cav), qeye(self.N_qb))
    numerator = expect(a.dag()**n * a**n, rho)
    denom = (expect(a.dag() * a, rho))**n
    return numerator/denom


# Each H test for fock states; steady state for first three H's; Optimal qubit frequency
