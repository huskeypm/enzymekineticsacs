# 
import matplotlib.pylab as plt
class empty:pass

print "WARNING: should test time-dependent soln of code against analytical results" 


# TODO 
# use Johan'vA2 meshes
# add in Johan'vA2 fluxes
# add back reactions 
# add right hand compart (remember, need to have more generall way of defining weak forms for all these rxns 


# Units 
# Distances [um] 
# Concentrations [uM]
# Time [ms] 
# Diff constants [um^2/ms]  
paraview=False 
verbose=False 
idxA1 = 0 # PDE 
idxA2 = 2 # left compartment 
idxA3 = 4 # right compartment 


## Units
nm_to_um = 1.e-3

print "WARNING: fixunits" 
Ds=0.01     # very slow [um^2/ms]
Dw=1.       # water [um^2/ms]
Df = 1000.


from dolfin import *
import numpy as np

import cPickle as pickle
def writepickle(fileName,ts,concs,vars=-1,verbose=False):
  # store 
  data1 = {'ts':ts,'concs':concs,'vars':vars}

  if verbose:
    print "Writing ", fileName
  output = open(fileName, 'wb')

  # Pickle dictionary using protocol 0.
  pickle.dump(data1, output)

  output.close()

def readpickle(fileName,novar=False):    
  pkl_file = open(fileName, 'rb')
  data1 = pickle.load(pkl_file)
  ts  = data1['ts']
  concs  = data1['concs']  
  if novar:
    vars = 1
  else: 
    vars   = data1['vars']  
  pkl_file.close()
  return ts,concs,vars



## My reaction system 
# dc/dt = 
# Volume '1' is our diffusional domain
# Volume '2' is our ode domain 

debug = False


## Params 
# compartments 
dist = 1.  # [um] PKH what is this - dist between 1/2 and 1/3 
dist = 1.*nm_to_um  # [um] PKH what is this - dist between 1/2 and 1/3 
dist = Constant(dist)  # [nm?] PKH what is this - dist between 1/2 and 1/3 

# kinetics 
kp = 1.0     
km = 0.6 
bT = 70.0   # [uM]  
Kd = km/kp; # [uM] 

# concentrations 
nComp = 6



class Params():
  # time steps 
  steps = 500
  dt = 1.0   # [ms] 
  steps = 250
  dt = 2.0
  

  # diffusion params 
  D1   = 1.  # [um^2/ms] Diff const within PDE (domain 1) 
  D12  = 1000.  # [um^2/ms] Diff const between domain 1 and 2
  D13  = 1000.  # [um^2/ms] Diff const between domain 1 and 3

  # init concs 
  cA1init = 0.5 # [uM]
  cB1init = 0.5 # [uM]
  cA2init =1.0
  cB2init =1.0
  cA3init =0.1
  cB3init =0.1

  # buffer (PDE domain for now) 
  cBuff1= 0. # concentration [uM]
  KDBuff1 = 1. # KD [uM]  

  # source
  periodicSource=False # periodic source on CA2
  amp = 0.05 # [uM] 
  freq = 0.1 # [kHz]? 

  # geometry 
  #np.max(mesh.coordinates()[:],axis=0) - np.min(mesh.coordinates()[:],axis=0) 
  meshDim = np.array([100.,100.,100.])*nm_to_um # [um] 
  volume_scalar2  = 1. # [um^3] 
  volume_scalar3  = 1. # [um^3] 




class InitialConditions(Expression):

  def eval(self, values, x):
    # edge  
    #if (x[0] < 0.5 and np.linalg.norm(x[1:2]) < 0.5):
    # corner 
    #oif (np.linalg.norm(x -np.zeros(3) ) < 0.5):
    if 1:
      values[0] = self.params.cA1init
      values[1] = self.params.cB1init
      values[2] = self.params.cA2init
      values[3] = self.params.cB2init
      values[4] = self.params.cA3init
      values[5] = self.params.cB3init
    #else:
    #  values[0] = 0         
    #  values[1] = 0
    #  values[2] = 0
    #  values[3] = 0
  def value_shape(self):
    return (6,)

# Class for interfacing with the Newton solver
class MyEqn(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.reset_sparsity = True
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A, reset_sparsity=self.reset_sparsity)
        self.reset_sparsity = False

def Report(u_n,mesh,t,concs=-1,j=-1):

    # 
    #xTest = [5.0,0.5,0.5]
    #u = u_n.split()[0]
    #print "A1 at xTest: ", u(xTest)              
    for i,ele in enumerate(split(u_n)):
      tot = assemble(ele*dx,mesh=mesh)
      vol = assemble(Constant(1.)*dx,mesh=mesh)
      conc = tot/vol
      if verbose: 
        print "t=%f Conc(%d) %f " % (t,i,conc)
      if(j>-1): 
        concs[j,i] = conc

def PrintSlice(results): 
    mesh = results.mesh
    dims = np.max(mesh.coordinates(),axis=0) - np.min(mesh.coordinates(),axis=0)
    u = results.u_n.split()[0]
    up = project(u,FunctionSpace(mesh,"CG",1))
    res = 100
    (gx,gy,gz) = np.mgrid[0:dims[0]:(res*1j),
                          dims[1]/2.:dims[1]/2.:1j,
                          0:dims[2]:(res*1j)]
    from scipy.interpolate import griddata
    img0 = griddata(mesh.coordinates(),up.vector(),(gx,gy,gz))
    
    imgx=np.reshape(img0,[res,res])
    print np.shape(imgx)
    plt.pcolormesh(np.reshape(gx,[res,res]).T,np.reshape(gz,[res,res]).T,imgx.T,
           cmap=plt.cm.RdBu_r)
    plt.xlabel("x [um]")
    plt.ylabel("z [um]")
    plt.colorbar()

      

def Problem(params = Params()): 

  # get effective diffusion constant based on buffer 
  D1eff = params.D1 / (1 + params.cBuff1/params.KDBuff1)
  print "D: %f Dwbuff: %f [um^2/ms]" % (params.D1,D1eff)
  print "dim [um]", params.meshDim

  steps = params.steps 

  # rescale diffusion consants 

  dt = Constant(params.dt)
  D1 = Constant(D1eff)   # diff within domain 1 
  D12 = Constant(params.D12) # diffusionb etween domain 1 and 2 
  D13 = Constant(params.D13)

  # Define mesh and function space 
  #mesh = UnitSquare(16,16)
  marker12 = 10 # boundary marker for domains 1->2
  marker13 = 11 # boundary marker for domains 1->3
  mesh = Mesh("cube.xml.gz")
  mesh.coordinates()[:]*= params.meshDim
  face_markers = MeshFunction("uint",mesh, "cube_face_markers.xml.gz")
  ds = Measure("ds")[face_markers]
      
  
  
  ##
  volumeDom1 = Constant(assemble(Constant(1.0)*dx,mesh=mesh))
  area = Constant(assemble(Constant(1.0)*ds(marker12),mesh=mesh))
  #was volume_scalar2 = Constant(4.*float(volumeDom1))
  volume_frac12 = volumeDom1/params.volume_scalar2 
  #was volume_scalar3 = Constant(4.*float(volumeDom1))
  volume_frac13 = volumeDom1/params.volume_scalar3
  print "Vol V1 %f V2 %f V3 %f [um^3]" % (volumeDom1,params.volume_scalar2,params.volume_scalar3)
  
  # functions 
  V = FunctionSpace(mesh,"CG",1)
  R = FunctionSpace(mesh,"R",0)
  ME = MixedFunctionSpace([V,V,R,R,R,R])
  
  # Trial and Test functions 
  du = TrialFunction(ME) 
  vA1,vB1,vA2,vB2,vA3,vB3  = TestFunctions(ME)
  
  # Define function
  u_n = Function(ME) # current soln
  u0 = Function(ME) # prev soln
  
  # split mixed functions
  cA1_n,cB1_n,cA2_n,cB2_n,cA3_n,cB3_n = split(u_n)
  cA1_0,cB1_0,cA2_0,cB2_0,cA3_0,cB3_0 = split(u0)
  
  # Init conts
  #init_cond = InitialConditions()
  init_cond = InitialConditions()
  init_cond.params = params
  u_n.interpolate(init_cond)
  u0.interpolate(init_cond)
  
  
  ## Weak forms for RHS  
  # See notetaker 121213 notes for details 
  
  # Diffusion
  #if(field==False):
  #  RHSA1 = -Dij*inner(grad(c),grad(vA1))*dx  
  #  RHSB1 = -Dij*inner(grad(cb),grad(vB1))*dx 
  #else:
  RHSA1 = -inner(D1*grad(cA1_n),grad(vA1))*dx  
  RHSB1 = -inner(D1*grad(cB1_n),grad(vB1))*dx 
  RHSA2 = Constant(0)*vA2*dx # for consistency
  RHSB2 = Constant(0)*vB2*dx
  RHSA3 = Constant(0)*vA3*dx # for consistency
  RHSB3 = Constant(0)*vB3*dx
  
  # Reaction: b + c --kp--> cb,  
  #           b + c <-km--- cb
  R = np.array([
    [-kp,km],     # vA2
    [kp,-km]      # p
    ])
  
  
  # operator splitting 
  rxn=False
  #opSplit=True # just turning off reaction vA2
  if(rxn):
    # no rxn in PDE part   
    #RHSA1 += (R[0,0]*(bT-cB1_n)*cA1_n*vA1 + R[0,1]*cB1_n*vA1)*dx
    #RHSB1 += (R[1,0]*(bT-cB1_n)*cA1_n*vB1 + R[1,1]*cB1_n*vB1)*dx
  
    RHSA2 += (R[0,0]*(bT-cB2_n)*cA2_n*vA2 + R[0,1]*cB2_n*vA2)*dx
    RHSB2 += (R[1,0]*(bT-cB2_n)*cA2_n*vB2 + R[1,1]*cB2_n*vB2)*dx
  
    RHSA3 += (R[0,0]*(bT-cB3_n)*cA3_n*vA3 + R[0,1]*cB3_n*vA3)*dx
    RHSB3 += (R[1,0]*(bT-cB3_n)*cA3_n*vB3 + R[1,1]*cB3_n*vB3)*dx

  # periodic source
  expr = Expression("a*sin(b*t)",a=params.amp,b=params.freq,t=0)
  if params.periodicSource:
    print "Adding periodic source" 
    #RHSA2 += expr*cA2_n*vA2*dx
    RHSA2 += expr*vA2*dx
  
  
  # Time derivative and diffusion of field species
  # (dc/dt) = RHS  --> c1-cA1_0 - dt * RHS = 0
  # WRONG LA1 = (c-cA1_0)c*vA1*dx - cA1_0*vA1*dx - dt * RHSA1
  #LB1 = cb*vB1*dx - cB1_0*vB1*dx - dt * RHSB1
  #LA2 = cs*vA2*dx - cA2_0*vA2*dx - dt * RHSA2
  #LB2 = ct*vB2*dx - cB2_0*vB2*dx - dt * RHSB2
  LA1 = (cA1_n-cA1_0)*vA1/dt*dx - RHSA1 
  LB1 = (cB1_n-cB1_0)*vB1/dt*dx - RHSB1 
  
  # Flux to mesh domain
  LA1 -= D12*(cA2_n-cA1_n)*vA1/dist*ds(marker12)
  LB1 -= D12*(cB2_n-cB1_n)*vB1/dist*ds(marker12)
  LA1 -= D13*(cA3_n-cA1_n)*vA1/dist*ds(marker13)
  LB1 -= D13*(cB3_n-cB1_n)*vB1/dist*ds(marker13)
  
  
  # Time derivative of scalar species and flux from scalar domain 
  # domain 2
  LA2 = (cA2_n-cA2_0)*vA2/(dt*volume_frac12)*dx - RHSA2
  LB2 = (cB2_n-cB2_0)*vB2/(dt*volume_frac12)*dx - RHSB2
  LA2 += D12*(cA2_n-cA1_n)*vA2/dist*ds(marker12)
  LB2 += D12*(cB2_n-cB1_n)*vB2/dist*ds(marker12)
  # domain 3
  LA3 = (cA3_n-cA3_0)*vA3/(dt*volume_frac13)*dx - RHSA3
  LB3 = (cB3_n-cB3_0)*vB3/(dt*volume_frac13)*dx - RHSB3
  LA3 += D13*(cA3_n-cA1_n)*vA3/dist*ds(marker13)
  LB3 += D13*(cB3_n-cB1_n)*vB3/dist*ds(marker13)
  
  # compbine
  L = LA1 + LB1 + LA2 + LB2 + LA3 + LB3
  
  # Compute directional derivative about u in the direction of du (Jacobian)
  # (for Newton iterations) 
  a = derivative(L, u_n, du)
  
  
  # Create nonlinear problem and Newton solver
  problem = MyEqn(a, L)
  solver = NewtonSolver("lu")
  solver.parameters["convergence_criterion"] = "incremental"
  solver.parameters["relative_tolerance"] = 1e-6
  
  # Output file
  file = File("output.pvd", "compressed")


  Report(u_n,mesh,0)                

  
  # Step in time
  t   = 0.0
  T = steps*float(dt)
  tots=np.zeros([steps,nComp])
  concs=np.zeros([steps,nComp])
  ts=np.zeros(steps)
  j=0
  
  while (t   < T):
      #print "t", t
      solver.solve(problem, u_n.vector())
  
      ## store 
      # check values
      Report(u_n,mesh,t,concs=concs,j=j)

      #for i,ele in enumerate(split(u_n)):
      #  tot = assemble(ele*dx,mesh=mesh)
      #  vol = assemble(Constant(1.)*dx,mesh=mesh)
      #  conc = tot/vol  
      #  concs[j,i] = conc
      #  print "t=%f Conc(%d) %f " % (t,i,conc)          
  
      tots[j,0] = assemble(cA1_n*dx) # PDE domain 
      tots[j,1] = assemble(cB1_n*dx)
      tots[j,2] = assemble(cA2_n/volume_frac12*dx) # scalar domain
      tots[j,3] = assemble(cB2_n/volume_frac12*dx)
      tots[j,4] = assemble(cA3_n/volume_frac13*dx)
      tots[j,5] = assemble(cB3_n/volume_frac13*dx)
      ts[j] = t    

      if paraview:     
        file << (u_n.split()[0], t)    

      ## update 
      #file << (u,vB2)
      j+=1
      t   += float(dt)
      expr.t = t 
      u0.vector()[:] = u_n.vector()
  

  results = empty()
  results.ts = ts 
  results.concs = concs 
  results.tots = tots 
  results.mesh = mesh 
  results.u_n = u_n

  return results

def test12():
  ## test block of left channel 
  params = Params()
  params.D12=1000; params.D13=0. 
  result        = Problem(params=params)

  ## assert 
  # for 5 timesteps 
  #finalRef = np.array([  37.089814176,  37.089814176, 250.910186824, 250.910186824,25.6,25.6])
  finalRef = np.array([  0.7499539 ,0.7499539 ,0.7500461 ,0.7500461, 0.1 ,      0.1  ])           
  print "start", result.tots[0,:] 
  final = result.tots[-1,:] 
  print "final",final 
  for i in np.arange(nComp): 
    assert(np.abs(final[i] - finalRef[i])< 0.001), "Failed for species %d [%f/%f]" % (i,final[i],finalRef[i])

  return result

def test13():
  ## test block of right channel 
  params = Params()
  params.D12=0; params.D13=1000
  result = Problem(params=params)

  ## assert 
  # for 5 timesteps 
  #finalRef = np.array([  27.98166065 , 27.98166065 ,256.  ,       256.   ,       29.61833935,   29.61833935])
  finalRef = np.array([  0.30002065, 0.30002065, 1. ,        1.  ,       0.29997935, 0.29997935])                 
  final = result.tots[-1,:] 
  print final 
  for i in np.arange(nComp): 
    assert(np.abs(final[i] - finalRef[i])< 0.001), "Failed for species %d [%f/%f]" % (i,final[i],finalRef[i])

  return result

def test4():
  params = Params()
  Params.meshDim = np.array([1.,1.,1.])   # [um] 

  ## checked that domains remain constant 
  if 0: 
    params.cA1init=0.0000  
    params.D1=0.        
    params.cA2init=100.
    params.cA3init=0.0000
    params.D12=0.; params.D13=0.
    result        = Problem(params=params)

  ## conc even in all domains 
  if 0: 
    params.cA1init=0.0000  
    params.D1=1000.     
    params.cA2init=100.
    params.cA3init=0.0000
    params.D12=1000.; params.D13=1000.
    result        = Problem(params=params)

  ## conc even in all domains 
  if 1: 
    params.volume_scalar2  = 1000. # [um^3] 
    params.volume_scalar3  = 1000. # [um^3] 
    params.steps = 100
    params.dt = 1 # [ms] 
    params.cA1init=0.0000  
    params.D1=0.145 # [um^2/ms] DATP 
    params.cA2init=1.
    params.cA3init=0.0000
    params.D12=1000.; params.D13=1000.
    params.meshDim = np.array([10.,1.,1.])   # [um] 
    result        = Problem(params=params)

    
  return 
  ts = result.ts
  concs = result.concs
  plt.plot(ts,concs[:,0],label="A1/PDE")    
  plt.plot(ts,concs[:,2],label="A2")        
  plt.plot(ts,concs[:,4],label="A3")        
  plt.legend(loc=0)
  plt.gcf().savefig("x.png") 

def test5(arg=0):
  if(arg=="all" or arg==1): 
    test5i(Dbarrier=Ds,pickleName ="Dslow.pkl")
  if(arg=="all" or arg==2): 
    test5i(Dbarrier=Dw,pickleName ="Dwater.pkl")
  if(arg=="all" or arg==3): 
    test5i(Dbarrier=Df,pickleName ="Dfast.pkl")



# crappy way of doing this (should use inheritance)  
# oscillatory parameters 
def oscparams(Dbarrier=1.):
  amp = 0.05
  amp =  75  # gets to 3x original conc
  freq = 0.1
  Df = 1000.

  params = Params()
  params.cA2init=1.0
  params.cA1init=1.0
  params.cA3init=1.0  
  params.D12 = Df            
  params.D13 = Df            
  params.D1 = Dbarrier 
  params.amp=amp
  params.freq = freq
  params.periodicSource=True  
  params.meshDim = np.array([100.,100.,100.])*nm_to_um # [um]

  return params 


def test5i(Dbarrier=1.,pickleName="test.pkl"):

  steps = 500
  dt = 1.
  steps = 250
  dt = 2.
  # test
  #steps=10  
  #dt = 4
  

  # w diff barrier 
  params = oscparams(Dbarrier=Dbarrier)
  params.dt =dt
  params.steps = steps   

  results =  Problem(params=params)
  tsp = results.ts
  concsp = results.concs  
  writepickle(pickleName,tsp,concsp)
      

  ts, concs,vars= readpickle(pickleName)
  
  plt.title("Dbarrier %f [um^2/ms]" % Dbarrier)    
  plt.plot(ts,concs[:,0],label="A1/PDE")    
  plt.plot(ts,concs[:,2],label="A2")        
  plt.plot(ts,concs[:,4],label="A3")        
  plt.legend()
  plt.gcf().savefig("test5.png") 


def test6(arg="diffs"):
  ## reset for each compatment 
  params = oscparams()
  params.cBuff1 = 0. # conc buffer [uM] 
  
  steps = 250  
  params.steps = steps 
  params.dt = 2
  
  ## test
  

  nvars= 3
  if(arg=="diffs"):
    vars = 10**np.linspace(-1.,1,nvars) 
    pickleName ="Ddiffs.pkl"

  if(arg=="dists"):
    #vars = 10**np.linspace(1.8,2.2,nvars) 
    vars = 10**np.linspace(1,3,nvars) 
    pickleName ="Ddists.pkl"

  if(arg=="distsLag"):
    #vars = 10**np.linspace(1.8,2.2,nvars) 
    nvars = 10 
    vars = 10**np.linspace(1,2.5,nvars) 
    steps = 1000 
    params.dt = 0.5

    # debug 
    #nvars = 3
    #vars = 10**np.linspace(1,3,nvars) 

    params.steps = steps 
    pickleName ="Ddistslag.pkl"


  if(arg=="buffs"):
    #vars = np.linspace(0,100,nvars)    
    vars = 10**np.linspace(1,3,nvars) 
    pickleName ="Dbuffs.pkl"

  concArA1 = np.zeros([steps,nvars])  
  concArA2 = np.zeros([steps,nvars])  
  concArA3 = np.zeros([steps,nvars])  

  for j, var in enumerate(vars):
    if arg=="diffs":
        params.D1 = var  

    if (arg=="dists" or arg=="distsLag"):
        vol = 100.*100*100.
        yz = np.sqrt(vol/var)
        conservVol=True
        if conservVol:
          params.meshDim = np.array([var,yz,yz])*nm_to_um # [um]
        else: 
          params.meshDim = np.array([var,100.,100.])*nm_to_um # [um]
        print "mesh dim", params.meshDim

    if arg=="buffs":
        params.D1 = 1. # 
        params.cBuff1 = var  

    results = Problem(params=params)
    # check that totals are conserved 
    start = np.sum(results.tots[0,[idxA1,idxA2,idxA3]])
    final = np.sum(results.tots[-1,[idxA1,idxA2,idxA3]])
    print "Tot conc at start %f and end %f" %(start,final) 

    concsij = results.concs  
    tsij = results.ts  
    concArA1[:,j] = concsij[:,idxA1]
    concArA2[:,j] = concsij[:,idxA2]
    concArA3[:,j] = concsij[:,idxA3]


  concs =  [concArA1,concArA2,concArA3]       
  writepickle(pickleName,tsij,concs,vars=vars)
    
  tsij,concsr,vars = readpickle(pickleName)
  concArA1 = concsr[0]
  concArA2 = concsr[1]
  concArA3 = concsr[2]
    
  j=0   
  styles2=['k-','k-.','k.']
  styles3=['b-','b-.','b.']
  plt.plot(tsij,concArA2[:,j],styles2[j],label="A2, v: %3.1f [] " % vars[j])
  plt.plot(tsij,concArA3[:,j],styles3[j],label="A3, v: %3.1f [] " % vars[j])
  j=1    
  plt.plot(tsij,concArA2[:,j],styles2[j],label="A2, v: %3.1f [] " % vars[j])
  plt.plot(tsij,concArA3[:,j],styles3[j],label="A3, v: %3.1f [] " % vars[j])
  
  j=2    
  plt.plot(tsij,concArA2[:,j],styles2[j],label="A2, v: %3.1f [] " % vars[j])
  plt.plot(tsij,concArA3[:,j],styles3[j],label="A3, v: %3.1f [] " % vars[j])
  
  plt.legend(loc=2)    
  

  plt.gcf().savefig(arg+".png")   

def test7():
  #print phis
  #print dists
  nPhis = 11
  #nBuffs = 3
  nDists = 10 
    
  #KD = 1. # 1 [uM]     
  #phis = np.linspace(0.5,1.0,nPhis)
  #cBuffs = np.linspace(0,5,nBuffs) # [uM]
  dists = 10**np.linspace(1,3,nDists)
  Deffs = 10**np.linspace(-2,0,nPhis)
  #print KDs


  params = oscparams(Dbarrier=-1)
  steps =100
  dt = 5  

  params.steps=steps
  params.dt = dt
  vol = 100.*100*100.
  pickleName="allf.pkl"  
  stddevs = np.zeros([nDists,nPhis])

  for i,dist in enumerate(dists):
    # adj diffusion barrier length while conserving vol  
    #dist = 100  
    yz = np.sqrt(vol/dist)  
    
    for j, Deff in enumerate(Deffs): 
      msg =  "Running dist/Deff %f %f" % (dist,Deff)    
      print msg
      params.meshDim = np.array([dist,yz,yz])*nm_to_um # [um]  
      params.D1 = Deff
      results = Problem(params=params)
      ts = results.ts
      concs = results.concs
      
      #plt.figure()
      #plt.title(msg)
      #plt.plot(ts,concs[:,idxA2])
      #plt.plot(ts,concs[:,idxA3])
      stddevs[i,j] = np.std(concs[:,idxA3])/np.std(concs[:,idxA2])
      print "s%f \n"%(stddevs[i,j])
      #break 
    
 
    
  writepickle(pickleName,ts,stddevs)    

  plt.pcolormesh(Deffs,dists,stddevs,cmap="Greys_r")
  plt.xlabel("dists")
  plt.ylabel("Deffs") 
  plt.gcf().savefig("test7.png") 

  
    

  



def validation(): 
  Params.debug = True
  Params.dt = 1.
  Params.steps = 5
  Params.meshDim = np.array([1.,1.,1.])   # [um] 
  # based on 4x4 grid, need to recompute for 1x1 grid 
  test12()
  test13()






#
#validation()
#!/usr/bin/env python
import sys
#
# Revisions
#       10.08.10 inception
#

def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -validation" % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg

if __name__ == "__main__":
  import sys
  msg = helpmsg()
  remap = "none"
  verbose = True

  raise RuntimeError("Replaced by 3comp.py") 

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  fileIn= sys.argv[1]
  if(len(sys.argv)==3):
    print "arg"

  for i,arg in enumerate(sys.argv):
    if(arg=="-validation"):
      #arg1=sys.argv[i+1] 
      validation()
      quit()
    if(arg=="-test4"): 
      test4()
      quit()
    if(arg=="-test5"): 
      test5("all")
      quit()
    if(arg=="-test5i"): 
      test5(np.int(sys.argv[i+1]))
      quit()
    if(arg=="-test6"): 
      test6(sys.argv[i+1])
      quit()
    if(arg=="-test7"): 
      test7()#sys.argv[i+1])
      quit()
  





  raise RuntimeError("Arguments not understood")




