import numpy as np
cimport numpy as np
import time
from time import clock
from spinwaves.spinwavecalc.readfiles import atom, readFiles
from periodictable import elements


#cdef class simple_C_atom:
#    cdef np.ndarray[np.int32_t, ndim=1] neighbors, interactions
#    cdef np.ndarray[np.float64_t, ndim=1] spin
#    
#    def __init__(self, np.ndarray[np.int32_t, ndim=1] neighbors, np.ndarray[np.int32_t, ndim=1] interactions, np.ndarray[np.float64_t, ndim=1] spin):
#        self.neighbors = neighbors
#        self.interactions = interactions
#        self.spin = spin

#cdef struct simple_C_atom:
#    np.ndarray[np.int32_t, ndim=1] neighbors
#    #int[] neighbors
#    np.ndarray[np.int32_t, ndim=1] interactions
#    np.ndarray[np.float64_t, ndim=1] spin[3]

def mcQueeny_case(interactionfile, spinfile, tau_list, temperature, 
                      direction={'kx':1,'ky':0,'kz':0}, hkl_interval=[1e-3,2*np.pi,100], 
                      omega_interval=[0,5,100], eig_eps = 0.001):
                      
    cdef int j,i,n,Qindex
    cdef np.ndarray[np.int32_t, ndim=1] magCellSize
    
    mat_time = 0.0 #time spent producing the matrices
    eig_time = 0.0 #time spent finding eigenvalues
    norm_time = 0.0
    strfac_time = 0.0
    lorentzian_time = 0.0    
    initial_time = 0.0
    
    #to plot eigenvector components
#    q1 = []
#    q2 = []
    
    #to plot w's
    #w1_list = []
    #w2_list = []
    #w3_list = []
    #w4_list = []
    #w5_list = []
    #w6_list = []
    #w7_list = []
    #w8_list = []
#    w_output_list = []
    
    #plot the structure factor to compare with lovesey (In McQueeny Paper)
#    strfac_sums = []
#    strfac1 = []
#    strfac2 = []
    
    atom_list, jnums, jmats, N_atoms_uc, numCutOff, magCellSize = readFiles(interactionfile,spinfile, rtn_cutOffInfo = True)
    
    #convert the results from readFiles to cython static code
    #list = []
    #cdef np.ndarray[np.int32_t, ndim=1] nbrs, ints
    #cdef np.ndarray[np.float64_t, ndim=1] spin
    #for atom in range(atom_list):
        #cdef np.ndarray[np.int32_t, ndim=1] nbrs = np.array(atom.neighbors)
        #cdef np.ndarray[np.int32_t, ndim=1] ints = np.array(atom.interactions)
        #cdef np.ndarray[np.float64_t, ndim=1] spin = np.array(atom.spin)
        #list.append(simple_C_atom(atom.neighbors, atom.interactions, atom.spin))
    #cdef np.ndarray[simple_C_atom, ndim=1] c_atom_list = np.array(list)
    #Rather than keeping the same structure(which doesn't seem possible), I will just make 3 arrays:
    cdef np.ndarray[np.ndarray[np.int32,ndim=1], ndim = 1] neighbor_list

    h_list, k_list, l_list = generate_hkl(hkl_interval, direction)
    e = generate_wt(omega_interval)
    ff_list = generate_form_factors(N_atoms_uc, atom_list, hkl_interval)
    s = []
    basis = []
    initial_time = time.clock()
    for a in atom_list:
        s.append(a.spin[2])#z component of spin (assume aligned with z)                
        #basis.append(a.pos)
        basis.append(a.pos - magCellSize) #translate  back to 0,0,0
        #normalize? NO!
        #basis.append([a.pos[0] - int(a.pos[0]), a.pos[1]-int(a.pos[1]), a.pos[2] - int(a.pos[2])])
    #print "\n\nbasis vectors:\n"
    
    #mess with basis vectors and spin
    #basis[0] = [0.25,0,0]
    #basis[0] = [1.0,0,0]
    #tmp = s[0]
    #s[0] = s[1]
    #s[1] = tmp
    #for pos in basis:
    #    print pos
    #print "\n\nspins:\n"
    #for spin in s:
    #    print spin
        
    
    mesh = np.zeros((len(h_list), len(e)))

    #output q, w, and eigenvectors before and after normalization to compare with McQueeny's
#    f = open("/home/tom/Desktop/Python_Eigs.txt", 'w')
#    f.write("q w eigVec_before_norm T\n\n")
    
#    f2 = open("/home/tom/Desktop/Values.txt", 'w')
#    f2.write("Q w wvec (wvec norm) e^(-i*Q.d_i)\n\n")
    
    
    #create the T_ni matricies (eigenvector matrices)
    #This algorithm uses a differnt matrix:
    #evecs, mats = calc_eig_mats_numerically(Hsave, h_list,k_list,l_list)
    #eigvals = calc_eigs_numerically(Hsave, h_list, k_list, l_list)#This is innefficient, eigs already calculated and thrown away in calc_eig_mats_..
    #w_list = gen_eigs_with_sign(Hsave, h_list, k_list, l_list, 0.001)
    cdef np.ndarray[np.float64_t, ndim=1] Q
    for Qindex in range(len(h_list)):
        #Q = (np.array([h_list,k_list,l_list]).transpose())[Qindex]
        Q = np.array((h_list[Qindex], k_list[Qindex], l_list[Qindex]))
#        f2.write(str(Q))
        #CALL SPINWAVE(q,w,evec) - get w, evec for given q
        tmp = clock()
        M = create_matrix(atom_list, jnums, jmats, numCutOff, magCellSize, Q-np.array(tau_list))
        mat_time += clock()-tmp
        tmp = clock()
        w, evec = np.linalg.eig(M)
        eig_time += clock() - tmp
#        f2.write(" " + str(w))
#        f2.write(" " + str(evec.flatten()))
        
        #                evec = evecs[Qindex]
        #                w = w_list[Qindex]
        
        
        
        
      #  print "\nw = ", w
        #w1_list.append(w[0])
        #w2_list.append(w[1])
        #w3_list.append(w[1])
        #w4_list.append(w[1])
        #w5_list.append(w[1])
        #w6_list.append(w[1])
        #w7_list.append(w[1])
        #w8_list.append(w[1])
#        w_output_list.append(w)
        
        
        ff = ff_list[Qindex]
        
        #Get number of atoms
        numAtoms = numCutOff
        #numAtoms = np.sqrt(evec.size)#This should be the number of atoms that I need to loop through.
        #if int(numAtoms) != numAtoms:#I beleive t_mat should always be square, but lets check
        #    raise Exception("t_mat not square!")
        
        #Normalize evec to get T_ni
        #for i in range(0, numAtoms):
        #    normi = 0
        #    for j in range(0, numAtoms):
        #        sigma_j = s[j]/np.abs(s[j])
        #        normi += sigma_j * np.abs(evec[j][i])**2
        #    for j in range(0, numAtoms):
        #        evec[i][j] = evec[j][i]/np.sqrt(np.abs(normi))
        
        evec2 = evec.copy()
        #I'm going to write this from scratch
      #  print "evec before = ", evec
        T = evec
        sigma = []
        tmp = clock()
        for i in range(numAtoms):
            sigma.append(s[i]/np.abs(s[i]))
        for n in range(numAtoms):
            norm = 0
            for i in range(numAtoms):
                norm += sigma[i]*(np.abs(evec2[n][i])**2)
            for i in range(numAtoms):
      #          print "norm : ", norm
      #          print "w sign: ", np.sign(w[n])
                #T[n][i] = T[n][i]*np.sqrt((w[n]/np.abs(w[n]))/norm + 0j)
                T[n][i] = evec2[n][i]/np.sqrt(np.abs(norm))
        evec = T
        norm_time += clock()-tmp
#        f2.write(" " + str(T.flatten()))
        #evec = evec2
#        q1.append(evec[0][0])
#        q2.append(evec[0][0])
        
        #output eigs to file
#        f.write(str(Q))
#        f.write(" w:" + str(w))
#        f.write("\nmat: " + str(M))#the matrix for which the eigs were found
#        f.write("\nbefore_norm: " + str(evec2.flatten()))
#        f.write("\nafterNorm: " + str(evec.flatten()) + "\n\n")
        
        #evec = evec.transpose()
        #test = 0
        #for i in range(numAtoms):
        #    sigma_i = s[j]/np.abs(s[j])
        #    test += sigma_i * np.dot(evec[i],evec[i])
        #    print "sigma_i = ", sigma_i
        #print "\n\nsum{sigma_i * |T_ni|^2} = ", test, "\n\n"
        
        #                print "evec = " , evec
        #                print "sigma1: ", s[0]/np.abs(s[0])
        #                print "sigma2: ", s[1]/np.abs(s[1])
        #                print "sign lambda_0: ", (s[0]/np.abs(s[0]))*(evec[0][0]**2) + (s[1]/np.abs(s[1]))*(evec[0][1]**2)
        #                print "sign lambda_1: ", (s[0]/np.abs(s[0]))*(evec[1][0]**2) + (s[1]/np.abs(s[1]))*(evec[1][1]**2)
                        
        
        #using only positive eigenvalue
        #w = [w[0]]
        #evec = [evec[0]]
        #print "eigenvalues used: ", w
        #print "eigenvectors used: ", evec    
    
    
          
        strfac = np.zeros(numAtoms)
        
        #for testing:
        ff = 1
        tmp = clock()
        for i in range(numAtoms):
            dum = 0
            for j in range(numAtoms):
                dum += (s[j]/np.sqrt(np.abs(s[j]))) * ff * evec[j][i] #* np.exp(1.0j*
                        #(-Q[0]*basis[j][0] -Q[1]*basis[j][1] -Q[2]*basis[j][2]))
#                print "\n\n\nwritting to f2:\n\n\n", np.exp(1.0j*
#                        (-Q[0]*basis[j][0] -Q[1]*basis[j][1] -Q[2]*basis[j][2]))
#                print "j: ", j
#                print "basis: ", basis[j]
#                print "Q.di", (-Q[0]*basis[j][0] -Q[1]*basis[j][1] -Q[2]*basis[j][2])
                exp = np.exp(1.0j*(-Q[0]*basis[j][0] -Q[1]*basis[j][1] -Q[2]*basis[j][2]))
#                f2.write(" " + str(exp))
            strfac[i] = np.abs(dum)**2
        strfac_time += clock()-tmp
       #     print "Q: ", Q, "    strfac[",i,"]: ", strfac[i]
#            f2.write("\n\n")
#        strfac_sums.append(0.0);
#        strfac1.append(strfac[0])
#        strfac2.append(strfac[1])

        sqw = np.zeros(len(e))
        tmp = clock()
        for j in range(numAtoms):
#            strfac_sums[Qindex] +=strfac[j]
            for i in range(len(e)):
                y = LIFETIME_VALUE/2
                lorentzian = (1/np.pi)*y/((e[i]- w[j])**2+y**2)
                #test
                #strfac[j] = ff*1
                sqw[i] = sqw[i] + strfac[j] * lorentzian #he has some 'bose' factor in here      
                #more test
                #sqw[i] = sqw[i] + strfac[j]
        mesh[Qindex] = sqw #* 0.5*(1 + Q[0]**2) #along x, Q_hat is just 1?
        lorentzian_time += clock() - tmp

    #f.close()
    #f = open("/home/tom/Desktop/output.txt", 'w')
    #for q in range(0,len(h_list)):
        #for w in range(0,len(e)):
            #f.write(str(h_list[q]))
            #f.write(" ")
            #f.write(str(e[w]))
            #f.write(" ")
            #f.write(str(mesh[q][w]))
            #f.write("\n")
    #f.close()
    #f = open("/home/tom/Desktop/output2.txt", 'w')
    #for q in range(0,len(h_list)):
            #f.write(str(h_list[q]))
            #f.write(" ")
            #f.write(str(np.abs(q1[q])))
            #f.write(" ")
            #f.write(str(np.abs(q2[q])))
            #f.write("\n")
    #f.close()
    #f = open("/home/tom/Desktop/output3.txt", 'w')
    #for q in range(0,len(h_list)):
            #f.write(str(h_list[q]))
            #f.write(" ")
            #f.write(str(np.real(w1_list[q])))
            #f.write(" ")
            #f.write(str(np.real(w2_list[q])))
            #f.write(" ")
            #f.write(str(np.real(w3_list[q])))
            #f.write(" ")
            #f.write(str(np.real(w4_list[q])))
            #f.write(" ")
            #f.write(str(np.real(w5_list[q])))
            #f.write(" ")
            #f.write(str(np.real(w6_list[q])))
            #f.write(" ")
            #f.write(str(np.real(w7_list[q])))
            #f.write(" ")
            #f.write(str(np.real(w8_list[q])))
            #f.write("\n")
    #f.close()
    #f = open("/home/tom/Desktop/strfacs.txt", 'w')
    #for q in range(0,len(h_list)):
            #f.write(str(h_list[q]))
            #f.write(" ")
            #f.write(str(k_list[q]))
            #f.write(" ")
            #f.write(str(l_list[q]))
            #f.write(" ")
            #f.write(str(strfac1[q]))
            #f.write(" ")
            #f.write(str(strfac2[q]))
            #f.write(" ")
            #f.write(str(strfac_sums[q]))
            #f.write("\n")
    #f.close()
    #plt.plot(w1_list)
    #plt.plot(w2_list)
    #plt.show()
    total_time = time.clock()-initial_time
    print "Run Time: ", total_time, " seconds"
    print "time spent producing matrices: ", mat_time,  "seconds | frac = ", mat_time/total_time
    print "time spent calculating eigs: ", eig_time,  "seconds | frac = ", eig_time/total_time
    print "time spent normalizing matrices: ", norm_time,  "seconds | frac = ", norm_time/total_time
    print "time spent calculating strfacs: ", strfac_time,  "seconds | frac = ", strfac_time/total_time
    print "time spent calculating and multiplying by lorentzians: ", lorentzian_time,  "seconds | frac = ", lorentzian_time/total_time
    

def get_Jij(jnums, jmats, num):
    """Returns the 3x3 numpy array corresponding to the matrix number, num from an atoms list
    of interactions."""
    index = jnums.index(num)
    return np.float64(np.array(jmats[index])) #This would be a sympy Matrix if I didn't use the np wrapper
            
def create_matrix(atom_list, jnums, jmats, int numCutOff, np.ndarray[np.int32_t, ndim=1] magCellSize, np.ndarray[np.float64_t, ndim=1] q):
    """Using hte Saenz technique.  The matrix indeices, i and j refer to atoms in the magnetic
    unit cell, so M will be an N*N matrix, where N is the number of atoms in the magnetic cell.
    However, the l_vector is still in cyrstallographic unit cell units.
    
    -numCutOff is the number of atoms in the cutoff/interaction (or really magnetic, but we
    are assuming htey are the same) cell.
    -magCellSize is a length 3 np.array of the size of the magnetic cell in terms of crystallographic
    unit cells. ie. (1,1,1) for simple ferromagnet or (2,1,1) for simple antiferromagnetic chain along
    x direction.
    -q should also be a 3 element numpy array
    -atom_list, jums, and jmats come from readfiles.  atom_list must contain all atoms in the
    interaction/magnetic cell as well as atoms that they bond to.  Atoms in the magnetic cell
    should appear first in the list.
    """
    
    cdef int i, j
    
    min_diff = 1e-8#he threshold for equality between floats(positions of atoms)
    #spin component to use:
    spin_comp = 2#use z component of spins (assume they are aligned along z)
    
    #Loop through the atoms in the cutoff cell?
    natoms = numCutOff #We really want the magnetic cell, but we are assuming thats the same as the interaction cell.
    #natoms = N_atoms_uc
    
    cdef np.ndarray[np.complex128_t, ndim=2] Mij = np.zeros((natoms,natoms), dtype = np.complex)
    
    #print "atoms in cell:"
    #for a in atom_list:
    #    print a.pos
    
    #Choose a unit cell to use as a reference point for the l vector
    #firstCellPos = atom_list[0].pos.astype(int)
    #print "first cell position: " , firstCellPos
    cdef int lk, l
    cdef np.float64_t spin_k, J, S_i, spin_j, sigma_j, S_j
    cdef np.complex128_t term_ij
    
    for i in range(natoms):
        for j in range(natoms):
            term_ij = 0.0 + 0j
            #print "i: ", i, "  j: ", j
            #print "atom_i pos: ", atom_list[i].pos
            if i == j:#delta_ij term (diaginal elements only)
                for lk in range(len(atom_list[i].neighbors)):
                    #assuming Jij matrix is diagonal, take one of the diagonal terms:
                    J = (get_Jij(jnums, jmats,atom_list[i].interactions[lk]))[1][1]
                    spin_k = atom_list[atom_list[i].neighbors[lk]].spin[spin_comp]
                    term_ij = term_ij + 2*J*spin_k#*sigma_k*S_k
            S_i = np.abs(atom_list[i].spin[spin_comp])
            spin_j = atom_list[j].spin[spin_comp]
            sigma_j = spin_j/np.abs(spin_j)
            S_j = np.abs(spin_j)
            #now for each atom correspondingto atom j in cell l sum J * exp(i*Q.l)
            #I'll loop through all the interactions, but only pick out those that bond
            #corresponding atoms in different(or the same) cells, ie. their positions - any integer
            #component are the same.
            atom_j = atom_list[j]
            for l in range(len(atom_list[i].neighbors)):
                atom_l = atom_list[atom_list[i].neighbors[l]]
                #Use unit cell lengtsh to construct l vecter
                l_pos = atom_l.pos
                #print "l_pos: ", l_pos
                #First we assumed it had to be a vector with integer components:
                #l_vec = l_pos.astype(int)-(atom_list[i].pos).astype(int)#firstCellPos
                #However, that ^ often produced imaginary numbers in the matrix
                l_vec = l_pos - atom_list[i].pos
                #use magnetic cell lengths to determine if atom_l is has the same position
                l_pos = l_pos/magCellSize
                l_pos = l_pos - l_pos.astype(int) #translate to first cell
                #in case l_pos is negative:
                l_pos = np.ceil(-l_pos) + l_pos
                #same for j_pos
                j_pos = atom_j.pos/magCellSize
                j_pos = j_pos - j_pos.astype(int)
                j_pos = np.ceil(-j_pos) + j_pos
                
                pos_diff = (l_pos)-(j_pos)#converted to magcell pos
                if ((np.abs(pos_diff)<min_diff).all()):
                    #print "l_vec: ", l_vec
                    J = (get_Jij(jnums, jmats, atom_list[i].interactions[l]))[1][1]#taking a value from diagonal matrix
                    #print "sinusoidal comp: ", 2*sigma_j*np.sqrt(S_i*S_j)*J*np.exp(1j*np.dot(q,l_vec))
                    term_ij = term_ij - 2*sigma_j*np.sqrt(S_i*S_j)*J*np.exp(1j*np.dot(q,l_vec))
                    #print "term[",i,"][",j,"]: ", term_ij
   #             elif (i != j):
                    #for testing:
    #                print ""
                    #print "\n\n\nOMG!!!  I shouldn't be here for the simple ferro/antiferro cases!\n\n\n"
            #print "term_ij: ", term_ij
            Mij[i][j] = term_ij
            #print "Mij: ", Mij
    #print "Q: ", q
    #print "M:\n", Mij
    #this creates eigenvalues that are different from ours by a factor of 2.  For now I'll
    #I'll just divide that out.
    return Mij/2.0




#The following functions were copied from csection_calc, but also left in that file becuase they are useed by both
#the MyQueeny algorithm and ours.  I duplicated the code becuase the code in this file is intended to be converted to Cython
#and I assume that talking between cython and pthon is slow.




#A copy of this was also left in cesction-calc because I assume talking between python and cyton is slow
def generate_hkl(hkl_interval, direction):
    
    #TODO - THIS DOES NOT WORK FOR ALL CASES! only when directional components are the same or 0
    #Fixed - testing
    
    #What if other points are 0 (not the first or last points)?
    #For that to happen they would need to generate an odd number of points across an interval -x to x
    if hkl_interval[0] == 0:
        hkl_interval[0] = 1e-3
    elif hkl_interval[1] == 0:
        hkl_interval[1] = -1e-3
        
    print "hkl_interval: ", hkl_interval
    print "direction: ", direction
    
    #We should be multipliying these linspaces by the components, kx,ky,kz, where those are
    #normalized so that kx**2 + ky**2 + kz**2 = 1 (assuming that h**2+k**2+l**2=q**2), I'm not
    #entirely sure if this works in inverse space.
    norm = np.sqrt(direction['kx']**2 + direction['ky']**2 + direction['kz']**2)

    if direction['kx']:
        h_list = np.linspace(hkl_interval[0],hkl_interval[1],hkl_interval[2])*direction['kx']/norm
    else:
        h_list = np.zeros(hkl_interval[2])
    if direction['ky']:
        k_list = np.linspace(hkl_interval[0],hkl_interval[1],hkl_interval[2])*direction['ky']/norm
    else:
        k_list = np.zeros(hkl_interval[2])
    if direction['kz']:
        l_list = np.linspace(hkl_interval[0],hkl_interval[1],hkl_interval[2])*direction['kz']/norm
    else:
        l_list = np.zeros(hkl_interval[2])
        
    #print "h,k,l points: "
    #for i in range(len(h_list)):
    #    print "(",h_list[i],", ", k_list[i], ", ", l_list[i], ")"

    return h_list, k_list, l_list

#A copy of this was also left in cesction-calc because I assume talking between python and cyton is slow
def generate_wt(omega_interval):

    return np.linspace(omega_interval[0],omega_interval[1],omega_interval[2])

#A copy of this was also left in cesction-calc because I assume talking between python and cyton is slow
def generate_form_factors(N_atoms_uc, atom_list, hkl_interval):
    # Form Factor
    eval_pnts = np.linspace(hkl_interval[0],hkl_interval[1],hkl_interval[2])
    ff_list = []
    for i in range(N_atoms_uc):
        try:
            elSym = atom_list[i].label
            elSym = elements.symbol(elSym)
            elMass = atom_list[i].massNum
            if elMass == None: #The mass number will now generally be None -Tom 1/28/11
                el = elSym
            else:
                el = elSym[elMass]
            print el
            val = atom_list[i].valence
            if val != None:
                Mq = el.magnetic_ff[val].M_Q(eval_pnts)
            else:
                Mq = el.magnetic_ff[0].M_Q(eval_pnts)
        except:
            Mq = np.zeros(len(eval_pnts))
        #Test to see how things look without ff
        #Mq = np.ones(len(eval_pnts))
        
        ff_list = Mq
#    fake = np.ones(len(kaprange))
    return np.array(ff_list)

from csection_calc import LIFETIME_VALUE