
from analysis import *
import scipy
import matplotlib.pylab as plt
from plotting import *
import scipy
import scipy.integrate
import testrxn
class empty:pass

# TODO 
# DONE might need to make this compatible wit a cube 
# use Johan'vAl meshes???
# add in Johan'vAl fluxes???
# add back reactions [DONE]  
# add right hand compart [DONE]


# Units 
# Distances [um] 
# Concentrations [uM]
# Time [ms] 
# Diff constants [um^2/ms]  

## WARNING: if changed, must also be reflected in plotting.py
idxAb = 0 # PDE 
idxBb = 1
idxCb = 2
idxAl = 3 # L compartment 
idxBl = 4
idxCl = 5
idxAr = 6 # R compartment 
idxBr = 7
idxCr = 8
nComp = 3
nSpec = 3
nDOF=nComp*nSpec

#noC = True # do not include 'C' DOFs in PDE solution. 


## Units
nm_to_um = 1.e-3



from dolfin import *
import numpy as np

import cPickle as pickle
def writepickle(fileName,ts,concs,vars=-1,lines=-1,verbose=False):
  # store 
  data1 = {'ts':ts,'concs':concs,'vars':vars, 'lines':lines}

  if verbose:
    print "Writing ", fileName
  output = open(fileName, 'wb')

  # Pickle dictionary using protocol 0.
  pickle.dump(data1, output)

  output.close()

def readpickle(fileName,novar=False,lines=False):    
  pkl_file = open(fileName, 'rb')
  data1 = pickle.load(pkl_file)
  ts  = data1['ts']
  concs  = data1['concs']  
  if novar:
    vars = 1
  else: 
    vars   = data1['vars']  

  dlines = data1['lines']
  pkl_file.close()

  print lines
  if lines==True:   
    return ts,concs,vars,dlines
  else: 
    return ts,concs,vars



## My reaction system 
# dc/dt = 
# Volume '1' is our diffusional domain
# Volume '2' is our ode domain 

debug = False


## Params 
# dist between ODE/PDE compartments 
dist = 1.*nm_to_um  # [um] PKH what is this - dist between 1/2 and 1/3 
dist = Constant(dist)  # 

class Params():
  paraview = False 
  verbose=False 

  # time steps 
  steps = 500
  dt = 1.0   # [ms] 
  steps = 250
  dt = 2.0

  Ds=0.100   # very slow [um^2/ms]
  Dbulk=1.       # water [um^2/ms]
  Df = 1e3   
  

  # diffusion params 
  DAb   = Dbulk# [um^2/ms] Diff const within PDE (domain 1) 
  DBb   = Dbulk    # [um^2/ms] Diff const within PDE (domain 1) 
  DCb   = Dbulk    # [um^2/ms] Diff const within PDE (domain 1) 
  D12  = Df    # [um^2/ms] Diff const between domain 1 and 2
  D13  = Df    # [um^2/ms] Diff const between domain 1 and 3

  # init concs 
  cAbinit =1.0  # [uM]
  cBbinit =1.0  # [uM]
  cCbinit =1.0  # [uM]
  cAlinit =1.0
  cBlinit =1.0
  cClinit =1.0
  cArinit =1.0 
  cBrinit =1.0 
  cCrinit =1.0 

  # buffer (PDE domain for now) 
  cBuff1= 0. # concentration [uM]
  KDBuff1 = 1. # KD [uM]  

  # source
  periodicSource=False # periodic source on CAl
  amp = 0.05 # [uM] 
  freq = 0.1 # [kHz]? 

  # reaction
  goodwinReaction = False
  Km = 1.
  p = 2.  #12.
  v0 = 5.  #360.
  k0 = 5.
  v1 = 1.
  k1 = 1. 
  v2 = 1.
  k2 = 1. 
  v3 = 1.
  k3 = 1. 

  # geometry 
  #np.max(mesh.coordinates()[:],axis=0) - np.min(mesh.coordinates()[:],axis=0) 
  meshDim = np.array([100.,100.,100.])*nm_to_um # [um] 
  volume_scalar2  = 1. # [um^3] 
  volume_scalar3  = 1. # [um^3] 
  meshName = "cube" # production




class InitialConditions(Expression):

  def eval(self, values, x):
    # edge  
    #if (x[0] < 0.5 and np.linalg.norm(x[1:2]) < 0.5):
    # corner 
    #oif (np.linalg.norm(x -np.zeros(3) ) < 0.5):
    if 1:
      values[0] = self.params.cAbinit
      values[1] = self.params.cBbinit
      values[2] = self.params.cCbinit
      values[3] = self.params.cAlinit
      values[4] = self.params.cBlinit
      values[5] = self.params.cClinit
      values[6] = self.params.cArinit
      values[7] = self.params.cBrinit
      values[8] = self.params.cCrinit
  def value_shape(self):
    return (nDOF,)

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

def Report(u_n,mesh,t,concs=-1,j=-1,params=False):   

    # 
    #xTest = [5.0,0.5,0.5]
    #u = u_n.split()[0]
    #print "Ab at xTest: ", u(xTest)              
    for i,ele in enumerate(split(u_n)):
      tot = assemble(ele*dx,mesh=mesh)
      vol = assemble(Constant(1.)*dx,mesh=mesh)
      conc = tot/vol
      if params and params.verbose: 
        print "t=%f Conc(%d) %f " % (t,i,conc)
      if(j>-1): 
        concs[j,i] = conc

def PrintLine(results,species="A"):
    if species=="A":
      img = PrintSlice(results,doplot=False,idx=idxAb)
    elif species=="B":
      img = PrintSlice(results,doplot=False,idx=idxBb)
    elif species=="C":
      img = PrintSlice(results,doplot=False,idx=idxCb)
    s = np.shape(img)

    line = img[:,int(s[1]/2.)]
    #print np.mean(line) 
    #print line[0],line[-1]
    return line 

def PrintSlice(results,doplot=True,idx=idxAb): 
    mesh = results.mesh
    dims = np.max(mesh.coordinates(),axis=0) - np.min(mesh.coordinates(),axis=0)
    u = results.u_n.split()[idx]
    up = project(u,FunctionSpace(mesh,"CG",1))
    res = 100
    (gx,gy,gz) = np.mgrid[0:dims[0]:(res*1j),
                          dims[1]/2.:dims[1]/2.:1j,
                          0:dims[2]:(res*1j)]
    from scipy.interpolate import griddata
    img0 = griddata(mesh.coordinates(),up.vector(),(gx,gy,gz))
    
    imgx=np.reshape(img0,[res,res])

    if doplot:
      print np.shape(imgx)
      plt.pcolormesh(np.reshape(gx,[res,res]).T,np.reshape(gz,[res,res]).T,imgx.T,
             cmap=plt.cm.RdBu_r)
      plt.xlabel("x [um]")
      plt.ylabel("z [um]")
      plt.colorbar()

    return imgx

      

def Problem(params = Params()):

  # get effective diffusion constant (for C only) based on buffer 
  DAbeff = params.DAb 
  DBbeff = params.DBb 
  DCbeff = params.DCb / (1 + params.cBuff1/params.KDBuff1)
  print "D: %f Dbulkbuff: %f [um^2/ms]" % (params.DCb,DCbeff)
  print "dim [um]", params.meshDim

  steps = params.steps 

  # rescale diffusion consants 

  dt = Constant(params.dt)
  DAb = Constant(DAbeff)   # diff within domain 1 
  DBb = Constant(DBbeff)   # diff within domain 1 
  DCb = Constant(DCbeff)   # diff within domain 1 
  D12 = Constant(params.D12) # diffusionb etween domain 1 and 2 
  D13 = Constant(params.D13)

  # Define mesh and function space 
  #mesh = UnitSquare(16,16)
  marker12 = 10 # boundary marker for domains 1->2
  marker13 = 11 # boundary marker for domains 1->3
  mesh = Mesh(params.meshName+".xml.gz")
  mesh.coordinates()[:]*= params.meshDim
  face_markers = MeshFunction("uint",mesh, params.meshName+"_face_markers.xml.gz")
  ds = Measure("ds")[face_markers]
  
      
  
  
  ##
  volumeDom1 = Constant(assemble(Constant(1.0)*dx,mesh=mesh))
  params.volumeDom1 = volumeDom1
  area = Constant(assemble(Constant(1.0)*ds(marker12),mesh=mesh))
  #was volume_scalar2 = Constant(4.*float(volumeDom1))
  volume_frac12 = volumeDom1/params.volume_scalar2 
  #was volume_scalar3 = Constant(4.*float(volumeDom1))
  volume_frac13 = volumeDom1/params.volume_scalar3
  print "Vol V1 %f V2 %f V3 %f [um^3]" % (volumeDom1,params.volume_scalar2,params.volume_scalar3)
  
  # functions 
  V = FunctionSpace(mesh,"CG",1)
  R = FunctionSpace(mesh,"R",0)
  ME = MixedFunctionSpace([V,V,V,R,R,R,R,R,R])
  
  # Trial and Test functions 
  du = TrialFunction(ME) 
  vAb,vBb,vCb,\
  vAl,vBl,vCl,\
  vAr,vBr,vCr  = TestFunctions(ME)
  
  # Define function
  u_n = Function(ME) # current soln
  u0 = Function(ME) # prev soln
  
  # split mixed functions
  cAb_n,cBb_n,cCb_n,\
  cAl_n,cBl_n,cCl_n,\
  cAr_n,cBr_n,cCr_n = split(u_n)
  cAb_0,cBb_0,cCb_0,\
  cAl_0,cBl_0,cCl_0,\
  cAr_0,cBr_0,cCr_0 = split(u0)
  
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
  #  RHSAb = -Dij*inner(grad(c),grad(vAb))*dx  
  #  RHSBb = -Dij*inner(grad(cb),grad(vBb))*dx 
  #else:
  RHSAb = -inner(DAb*grad(cAb_n),grad(vAb))*dx  
  RHSBb = -inner(DBb*grad(cBb_n),grad(vBb))*dx 
  RHSCb = -inner(DCb*grad(cCb_n),grad(vCb))*dx 
  RHSAl = Constant(0)*vAl*dx # for consistency
  RHSBl = Constant(0)*vBl*dx
  RHSCl = Constant(0)*vCl*dx
  RHSAr = Constant(0)*vAr*dx # for consistency
  RHSBr = Constant(0)*vBr*dx
  RHSCr = Constant(0)*vCr*dx
  
  
  # operator splitting 
  #opSplit=True # just turning off reaction vAl
  if(params.goodwinReaction=="explicit"):
    # no rxn in PDE part   
    #RHSAb 
    #RHSBb 

    raise RuntimeError("Broken, since using new var names. See goodwin.py")
    # dA/dt =v0 / (1+ (C/K)^p) - k1*A
    # dB/dt =v1*A -k2*B
    # dC/dt =v2*B -k2*C

    p = params
    #  ->A
    ikm = 1/p.Km
    #RHSAl +=  (1/volume_frac12)*(p.v0/(1+(ikm*cCl_n)**p.p))*vAl*dx
    m = ikm*cCl_n
    m = m**p.p
    #m = m*m*m * m*m*m * m*m*m * m*m*m 
    RHSAl +=  (1/volume_frac12)*(p.v0/(1+m))*vAl*dx
 
    # A->B 
    # dA/dt = -k0*A
    # dB/dt = +v1*B
    RHSAl += -(1/volume_frac12)*p.k0*cAl_n*vAl*dx
    RHSBl +=  (1/volume_frac12)*p.v1*cAl_n*vBl*dx

    # B->C 
    # dB/dt = -k1*B
    # dC/dt = +v2*C
    RHSBl += -(1/volume_frac12)*p.k1*cBl_n*vBl*dx
    RHSCl +=  (1/volume_frac12)*p.v2*cBl_n*vCl*dx

    # C-> 0 
    RHSCl += -(1/volume_frac12)*p.k2*cCl_n*vCl*dx


  ## get indices/values (needed for operator splitting) 
  indcAb, indcBb, indcCb = set(), set(), set()
  indcAl, indcBl, indcCl = set(), set(), set()
  indcAr, indcBr, indcCr = set(), set(), set()
  dm_cAb, dm_cBb, dm_cCb = ME.sub(idxAb).dofmap(), ME.sub(idxBb).dofmap(),ME.sub(idxCb).dofmap()
  dm_cAl, dm_cBl, dm_cCl = ME.sub(idxAl).dofmap(), ME.sub(idxBl).dofmap(),ME.sub(idxCl).dofmap()
  dm_cAr, dm_cBr, dm_cCr = ME.sub(idxAr).dofmap(), ME.sub(idxBr).dofmap(),ME.sub(idxCr).dofmap()
  for cell in cells(mesh):
      indcAb.update(dm_cAb.cell_dofs(cell.index())); indcBb.update(dm_cBb.cell_dofs(cell.index())); indcCb.update(dm_cCb.cell_dofs(cell.index()))
      indcAl.update(dm_cAl.cell_dofs(cell.index())); indcBl.update(dm_cBl.cell_dofs(cell.index())); indcCl.update(dm_cCl.cell_dofs(cell.index()))
      indcAr.update(dm_cAr.cell_dofs(cell.index())); indcBr.update(dm_cBr.cell_dofs(cell.index())); indcCr.update(dm_cCr.cell_dofs(cell.index()))

  indcAb= np.fromiter(indcAb, dtype=np.uintc); indcBb= np.fromiter(indcBb, dtype=np.uintc); indcCb= np.fromiter(indcCb, dtype=np.uintc)
  indcAl= np.fromiter(indcAl, dtype=np.uintc); indcBl= np.fromiter(indcBl, dtype=np.uintc); indcCl= np.fromiter(indcCl, dtype=np.uintc)
  indcAr= np.fromiter(indcAr, dtype=np.uintc); indcBr= np.fromiter(indcBr, dtype=np.uintc); indcCr= np.fromiter(indcCr, dtype=np.uintc)


    
  
  # periodic source
  expr = Expression("a*sin(b*t)",a=params.amp,b=params.freq,t=0)
  if params.periodicSource:
    print "Adding periodic source" 
    #RHSAl += expr*cAl_n*vAl*dx
    RHSAl += expr*vAl*dx
  
  
  # Time derivative and diffusion of field species
  # (dc/dt) = RHS  --> c1-cAb_0 - dt * RHS = 0
  # WRONG LAb = (c-cAb_0)c*vAb*dx - cAb_0*vAb*dx - dt * RHSAb
  #LBb = cb*vBb*dx - cBb_0*vBb*dx - dt * RHSBb
  #LAl = cs*vAl*dx - cAl_0*vAl*dx - dt * RHSAl
  #LBl = ct*vBl*dx - cBl_0*vBl*dx - dt * RHSBl
  LAb = (cAb_n-cAb_0)*vAb/dt*dx - RHSAb 
  LBb = (cBb_n-cBb_0)*vBb/dt*dx - RHSBb 
  LCb = (cCb_n-cCb_0)*vCb/dt*dx - RHSCb 
  
  # Flux to mesh domain
  LAb -= D12*(cAl_n-cAb_n)*vAb/dist*ds(marker12)
  LBb -= D12*(cBl_n-cBb_n)*vBb/dist*ds(marker12)
  LCb -= D12*(cCl_n-cCb_n)*vCb/dist*ds(marker12)
  LAb -= D13*(cAr_n-cAb_n)*vAb/dist*ds(marker13)
  LBb -= D13*(cBr_n-cBb_n)*vBb/dist*ds(marker13)
  LCb -= D13*(cCr_n-cCb_n)*vCb/dist*ds(marker13)
  
  
  # Time derivative of scalar species and flux from scalar domain 
  # domain 2
  LAl = (cAl_n-cAl_0)*vAl/(dt*volume_frac12)*dx - RHSAl
  LBl = (cBl_n-cBl_0)*vBl/(dt*volume_frac12)*dx - RHSBl
  LCl = (cCl_n-cCl_0)*vCl/(dt*volume_frac12)*dx - RHSCl
  LAl += D12*(cAl_n-cAb_n)*vAl/dist*ds(marker12)
  LBl += D12*(cBl_n-cBb_n)*vBl/dist*ds(marker12)
  LCl += D12*(cCl_n-cCb_n)*vCl/dist*ds(marker12)

  # domain 3
  LAr = (cAr_n-cAr_0)*vAr/(dt*volume_frac13)*dx - RHSAr
  LBr = (cBr_n-cBr_0)*vBr/(dt*volume_frac13)*dx - RHSBr
  LCr = (cCr_n-cCr_0)*vCr/(dt*volume_frac13)*dx - RHSCr
  LAr += D13*(cAr_n-cAb_n)*vAr/dist*ds(marker13)
  LBr += D13*(cBr_n-cBb_n)*vBr/dist*ds(marker13)
  LCr += D13*(cCr_n-cCb_n)*vCr/dist*ds(marker13)

  
  # compbine
  L = LAb + LBb + LAl + LBl + LAr + LBr
  L+= LCb + LCl + LCr
  
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


  Report(u_n,mesh,0,params=params)                

  
  # Step in time
  ti   = 0.0
  t = ti
  T = steps*float(dt)
  lines=[]
  linesAr = empty()
  linesAr.A=[]; linesAr.B=[]; linesAr.C=[]
  tots=np.zeros([steps+2,nDOF])
  concs=np.zeros([steps+2,nDOF])
  ts=np.zeros(steps+2)
  j=0

  # entire simulation interval
#  print "put elsewhere" 
#  from goodwin import *
#  tis = scipy.linspace(ti,T,steps)
#  y0s = np.array([params.cAbinit,params.cAbinit,params.cAbinit])
#  p = params
#  ks = np.array([1.0,1/p.Km,p.p,p.v0,p.k0,p.v1,p.k1,p.v2,p.k2])            
#  yTs = goodwinmodel(tis,y0s,ks)
  #print yTs[-1]
  #quit()
  
  
  results = empty()
  results.mesh = mesh 
  while (t  < T):
      j+=1
      t   += float(dt)
      #print "t", t
      solver.solve(problem, u_n.vector())

      # operator splitting 
      if(params.goodwinReaction=="opsplit"): # after solve? 
        u_array = u_n.vector().array() 
        tis = scipy.linspace(t,t+float(dt),steps)
        y0s = np.array([u_array[indcAl],u_array[indcBl], u_array[indcCl]])
        y0s = np.ndarray.flatten(y0s)
        #print y0s
        #print yTs
        yTs = goodwinmodel(tis,y0s,ks)
        yTs = yTs[-1] 
        u_array[indcAl]=yTs[0]; u_array[indcBl]=yTs[1];u_array[indcCl]=yTs[2];
        #print "%f %f %f" %(u_array[indcAl], u_array[indcBl],u_array[indcCl])
        u_n.vector()[:] = u_array

      # operator splitting 
      if(params.goodwinReaction=="opsplit2"): # after solve? 
        import goodwin 
        p = params
        u_array = u_n.vector().array() 
        tis = scipy.linspace(t,t+float(dt),steps)

        # left side 
        y0s = np.array([u_array[indcAl],u_array[indcBl], u_array[indcCl]])
        y0s = np.ndarray.flatten(y0s)
        #yTs = goodwinmodelComp1(tis,y0s,ks)
        ks = np.array([1.0,1/p.Km,p.p,p.v0,p.v1,p.v2,p.v3])
        yTs = goodwin.rxn(tis,y0s,ks)
        yTs = yTs[-1] 
        u_array[indcAl]=yTs[0]; u_array[indcBl]=yTs[1];u_array[indcCl]=yTs[2];

        # right side 
        y0s = np.array([u_array[indcAr],u_array[indcBr], u_array[indcCr]])
        y0s = np.ndarray.flatten(y0s)
        ks = np.array([1.0,1/p.Km,p.p,p.k0,p.k1,p.k2,p.k3])
        yTs = goodwin.rxn(tis,y0s,ks)
        #yTs = goodwinmodelComp3(tis,y0s,ks)
        yTs = yTs[-1] 
        u_array[indcAr]=yTs[0]; u_array[indcBr]=yTs[1];u_array[indcCr]=yTs[2];


        #print "%f %f %f" %(u_array[indcAl], u_array[indcBl],u_array[indcCl])
        u_n.vector()[:] = u_array

      # operator splitting 
      if(params.goodwinReaction=="opsplitTest"): # after solve? 
        import testrxn
        u_array = u_n.vector().array() 
        tis = scipy.linspace(t,t+float(dt),steps)

        # left side 
        y0s = np.array([u_array[indcAl],u_array[indcBl], u_array[indcCl]])
        y0s = np.ndarray.flatten(y0s)
        #yTs = goodwinmodelComp1(tis,y0s,ks)
        ks = [p.v0,p.v1,p.v2]
        #yTs = testrxn.rxnComp2(tis,y0s,ks)
        yTs = testrxn.rxn(tis,y0s,ks)
        yTs = yTs[-1] 
        u_array[indcAl]=yTs[0]; u_array[indcBl]=yTs[1];u_array[indcCl]=yTs[2];

        # right side 
        y0s = np.array([u_array[indcAr],u_array[indcBr], u_array[indcCr]])
        y0s = np.ndarray.flatten(y0s)
        #yTs = goodwinmodel(tis,y0s,ks)
        ks = [p.k0,p.k1,p.k2]
        #yTs = testrxn.rxnComp3(tis,y0s,ks)
        yTs = testrxn.rxn(tis,y0s,ks)
        yTs = yTs[-1] 
        #print yTs
        u_array[indcAr]=yTs[0]; u_array[indcBr]=yTs[1];u_array[indcCr]=yTs[2];


        #print "%f %f %f" %(u_array[indcAl], u_array[indcBl],u_array[indcCl])
        u_n.vector()[:] = u_array

  
      ## store 
      # check values
      Report(u_n,mesh,t,concs=concs,j=j,params=params)

      #for i,ele in enumerate(split(u_n)):
      #  tot = assemble(ele*dx,mesh=mesh)
      #  vol = assemble(Constant(1.)*dx,mesh=mesh)
      #  conc = tot/vol  
      #  concs[j,i] = conc
      #  print "t=%f Conc(%d) %f " % (t,i,conc)          
  
      tots[j,idxAb] = assemble(cAb_n*dx) # PDE domain 
      tots[j,idxBb] = assemble(cBb_n*dx)
      tots[j,idxCb] = assemble(cCb_n*dx)
      tots[j,idxAl] = assemble(cAl_n/volume_frac12*dx) # scalar domain
      tots[j,idxBl] = assemble(cBl_n/volume_frac12*dx)
      tots[j,idxCl] = assemble(cCl_n/volume_frac12*dx)
      tots[j,idxAr] = assemble(cAr_n/volume_frac13*dx)
      tots[j,idxBr] = assemble(cBr_n/volume_frac13*dx)
      tots[j,idxCr] = assemble(cCr_n/volume_frac13*dx)
      ts[j] = t    

      if params.paraview:     
        file << (u_n.split()[0], t)    
        results.u_n = u_n
        lines.append(PrintLine(results))
        linesAr.A.append(PrintLine(results,species="A"))
        linesAr.B.append(PrintLine(results,species="B"))
        linesAr.C.append(PrintLine(results,species="C"))

      ## update 
      #file << (u,vBl)
      expr.t = t 
      u0.vector()[:] = u_n.vector()
  

  results.ts = ts 
  results.concs = concs 
  results.tots = tots 
  results.u_n = u_n
  results.lines = lines 
  results.linesAr = linesAr
  import copy
  results.params = copy.copy(params)

  return results

def genFreqShifts(n=25,intv=5,steps=8001,dt=0.03,pklName = "freqshift.pkl"):
# old steps = 4001
# old dt = 0.05
#dt = 0.1 # Too low resolution to resolve freq
# ODE Ds = 10**np.linspace(-2,2-(1/n),n)
#Ds = 10**np.linspace(1,3-(1/n),n) 
  Ds = 10**np.linspace(1,3-(1/n),n) / 1000. # [um^2/ms]

  yts=[]

  #Ds = np.array([10,15,20,25])


  plt.figure()
  #n=1; Ds[0] = 1.0; fileName="test.png"
  nf = np.zeros(n)
  for i in np.arange(n):
    print Ds[i]

    #results, tode, yode = figA(steps =steps, dt = dt,doplot=0,DAb=1e3,DBb=1e3,DCb=Ds[i])
    #results, tode, yode = figA(steps =steps, dt = dt,doplot=0,DAb=1.,DBb=1.,DCb=Ds[i])
    results, tode, yode = figA(steps =steps, dt = dt,doplot=0,DAb=0.1,DBb=0.1,DCb=Ds[i],barrierLength=0.01)
    
    # limits (take latter 2/3s of trajectory(
    yt = results.concs[:,idxAb]
    l1,l2 = np.int(np.shape(yt)[0]/3.), np.shape(yt)[0]-2

    yt = results.concs[l1:l2,idxAl]
    t = results.ts[l1:l2]
    yts.append(yt)
    
    it,psd = freq(yt,t,oldstuff=False)
    freqMax = it[argmax(psd)] 
    nf[i] = freqMax
    print freqMax 
    if(i%intv==0):
      plt.plot(t,yt,label="DC=%f/f=%f Hz"%(Ds[i],freqMax))

  #plt.xlim(600,1000)    
  plt.legend(loc=0)
  plt.xlabel("t [ms]")
  
  plt.figure()
  freq(yt,t,doplot=True,oldstuff=False);
  #plt.xlim(0,.05)

  yts = np.asarray(yts)
  writepickle(pklName,yts,nf,vars=Ds)

  return t,yts    
    
    

def test12():
  ## test block of left channel 
  params = Params()
  params.steps=25
  params.D12=1000; params.D13=0. 
  params.cAbinit=0.5
  params.cBbinit=0.5
  params.cCbinit=0.5
  params.cAlinit=1.
  params.cBlinit=1.
  params.cClinit=1.
  params.cArinit=0.1
  params.cBrinit=0.1
  params.cCrinit=0.1
  result        = Problem(params=params)

  ## assert 
  # for 5 timesteps 
  #finalRef = np.array([  37.089814176,  37.089814176, 250.910186824, 250.910186824,25.6,25.6])
  finalRef = np.array([  0.7499539 ,0.7499539 ,0.749954,0.7500461 ,0.7500461, 0.7500461,0.1 ,      0.1,0.1  ])           
  print "start", result.tots[0,:] 
  final = result.tots[-1,:] 
  print "final",final 
  for i in np.arange(nDOF): 
    assert(np.abs(final[i] - finalRef[i])< 0.001), "Failed for species %d [%f/%f]" % (i,final[i],finalRef[i])

  return result

def test13():
  ## test block of right channel 
  params = Params()
  params.D12=0; params.D13=1000
  params.cAbinit=0.5
  params.cBbinit=0.5
  params.cCbinit=0.5
  params.cAlinit=1.
  params.cBlinit=1.
  params.cClinit=1.
  params.cArinit=0.1
  params.cBrinit=0.1
  params.cCrinit=0.1
  result = Problem(params=params)

  ## assert 
  # for 5 timesteps 
  #finalRef = np.array([  27.98166065 , 27.98166065 ,256.  ,       256.   ,       29.61833935,   29.61833935])
  finalRef = np.array([  0.30002065, 0.30002065,0.3000206, 1. ,        1.  ,1 ,    0.29997935, 0.29997935,0.29997935])                 
  final = result.tots[-1,:] 
  print final 
  for i in np.arange(nDOF): 
    assert(np.abs(final[i] - finalRef[i])< 0.001), "Failed for species %d [%f/%f]" % (i,final[i],finalRef[i])

  return result

def test4():
  params = Params()
  Params.meshDim = np.array([1.,1.,1.])   # [um] 

  ## checked that domains remain constant 
  if 0: 
    params.cAbinit=0.0000  
    params.DAb=0.        
    params.DBb=0.        
    params.DCb=0.        
    params.cAlinit=100.
    params.cArinit=0.0000
    params.D12=0.; params.D13=0.
    result        = Problem(params=params)

  ## conc even in all domains 
  if 0: 
    params.cAbinit=0.0000  
    params.DAb=1000.     
    params.DBb=1000.     
    params.DCb=1000.     
    params.cAlinit=100.
    params.cArinit=0.0000
    params.D12=1000.; params.D13=1000.
    result        = Problem(params=params)

  ## conc even in all domains 
  if 1: 
    params.volume_scalar2  = 1000. # [um^3] 
    params.volume_scalar3  = 1000. # [um^3] 
    params.steps = 100
    params.dt = 1 # [ms] 
    params.cAbinit=0.0000  
    params.DAb=0.145 # [um^2/ms] DATP 
    params.DBb=0.145 # [um^2/ms] DATP 
    params.DCb=0.145 # [um^2/ms] DATP 
    params.cAlinit=1.
    params.cArinit=0.0000
    params.D12=1000.; params.D13=1000.
    params.meshDim = np.array([10.,1.,1.])   # [um] 
    result        = Problem(params=params)

    
  return 
  ts = result.ts
  concs = result.concs
  plt.plot(ts,concs[:,idxAb],label="Ab/PDE")    
  plt.plot(ts,concs[:,idxAl],label="Al")        
  plt.plot(ts,concs[:,idxAr],label="Ar")        
  plt.legend(loc=0)
  plt.gcf().savefig("test4.png",dpi=300) 

def test5(arg=0):
  params = Params()
  if(arg=="all" or arg==1): 
    test5i(Dbarrier=params.Ds,pickleName ="Dslow.pkl",name="slow")
  if(arg=="all" or arg==2): 
    test5i(Dbarrier=params.Dbulk,pickleName ="Dwater.pkl",name="water")
  if(arg=="all" or arg==3): 
    test5i(Dbarrier=params.Df,pickleName ="Dfast.pkl",name="fast")



# crappy way of doing this (should use inheritance)  
# oscillatory parameters 
def oscparams(Dbarrier=1e3):
  amp = 0.05
  amp =  75  # gets to 3x original conc
  freq = 0.1

  params = Params()
  params.cAlinit=1.0
  params.cAbinit=1.0
  params.cArinit=1.0  
  params.D12 = params.Df            
  params.D13 = params.Df            
  params.DAb = Dbarrier 
  params.DBb = Dbarrier 
  params.DCb = Dbarrier 
  params.amp=amp
  params.freq = freq
  params.periodicSource=True  
  params.meshDim = np.array([100.,100.,100.])*nm_to_um # [um]

  return params 


def test5i(Dbarrier=1e3,pickleName="test.pkl",name=""):

  steps = 500
  dt = 1.
  steps = 250
  dt = 2.
  # test
  #steps=30  
  #dt = 5
  

  # w diff barrier 
  params = oscparams(Dbarrier=Dbarrier)
  params.dt =dt
  params.steps = steps   
  params.paraview = True 
  results =  Problem(params=params)
  tsp = results.ts
  concsp = results.concs  

  tsp = tsp[1:-1]
  concsp= concsp[1:-1]
  writepickle(pickleName,tsp,concsp,Dbarrier,lines=results.lines)
      

  ts, concs,vars= readpickle(pickleName)
  
  plt.title("Dbarrier %f [um^2/ms]" % Dbarrier)    
  plt.plot(ts,concs[:,idxAb],label="Ab/PDE")    
  plt.plot(ts,concs[:,idxAl],label="Al")        
  plt.plot(ts,concs[:,idxAr],label="Ar")        
  plt.legend()
  plt.gcf().savefig("test5"+name+".png",dpi=300) 


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
    steps = 2000 
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

  concArAb = np.zeros([steps,nvars])  
  concArAl = np.zeros([steps,nvars])  
  concArAr = np.zeros([steps,nvars])  

  for j, var in enumerate(vars):
    if arg=="diffs":
        params.DAb = var  
        params.DBb = var  
        params.DCb = var  

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
        params.DAb = 1. # 
        params.DBb = 1. # 
        params.DCb = 1. # 
        params.cBuff1 = var  

    results = Problem(params=params)
    # check that totals are conserved 
    start = np.sum(results.tots[0,[idxAb,idxAl,idxAr]])
    final = np.sum(results.tots[-1,[idxAb,idxAl,idxAr]])
    print "Tot conc at start %f and end %f" %(start,final) 

    concsij = results.concs  
    tsij = results.ts  
    concArAb[:,j] = concsij[:,idxAb]
    concArAl[:,j] = concsij[:,idxAl]
    concArAr[:,j] = concsij[:,idxAr]


  concs =  [concArAb,concArAl,concArAr]       
  writepickle(pickleName,tsij,concs,vars=vars)
    
  tsij,concsr,vars = readpickle(pickleName)
  concArAb = concsr[0]
  concArAl = concsr[1]
  concArAr = concsr[2]
    
  j=0   
  styles2=['k-','k-.','k.']
  styles3=['b-','b-.','b.']
  plt.plot(tsij,concArAl[:,j],styles2[j],label="Al, v: %3.1f [] " % vars[j])
  plt.plot(tsij,concArAr[:,j],styles3[j],label="Ar, v: %3.1f [] " % vars[j])
  j=1    
  plt.plot(tsij,concArAl[:,j],styles2[j],label="Al, v: %3.1f [] " % vars[j])
  plt.plot(tsij,concArAr[:,j],styles3[j],label="Ar, v: %3.1f [] " % vars[j])
  
  j=2    
  plt.plot(tsij,concArAl[:,j],styles2[j],label="Al, v: %3.1f [] " % vars[j])
  plt.plot(tsij,concArAr[:,j],styles3[j],label="Ar, v: %3.1f [] " % vars[j])
  
  plt.legend(loc=2)    
  

  plt.gcf().savefig(arg+".png",dpi=300)   

def test7(buff=False):
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
  steps =200
  dt = 2.5  

  # dbg
  dbg=False
  prefix=""
  if dbg:
    prefix="Dbg"
    nPhis = 1 
    nDists = 1
    dists = np.array([100])
    Deffs = np.array([0.1]) 
    steps = 100


  if buff:
    B = 1.
    KD = 1.
    Deffs = Deffs / (1 + B/KD)

  params.steps=steps
  params.dt = dt
  vol = 100.*100*100.

  if buff:
    pickleName=prefix+"allbuff.pkl"  
  else: 
    pickleName=prefix+"all.pkl"  

  stddevs = np.zeros([nDists,nPhis])

  for i,dist in enumerate(dists):
    # adj diffusion barrier length while conserving vol  
    #dist = 100  
    yz = np.sqrt(vol/dist)  
    
    for j, Deff in enumerate(Deffs): 
      msg =  "Running dist/Deff %f %f" % (dist,Deff)    
      print msg
      params.meshDim = np.array([dist,yz,yz])*nm_to_um # [um]  
      params.DAb = Deff
      params.DBb = Deff
      params.DCb = Deff
      results = Problem(params=params)
      ts = results.ts
      concs = results.concs
      
      #plt.figure()
      #plt.title(msg)
      #plt.plot(ts,concs[:,idxAl])
      #plt.plot(ts,concs[:,idxAr])
      discard = int(steps/5)
      concAr = concs[discard:-2,idxAr]
      concAl = concs[discard:-2,idxAl]
      stddevs[i,j] = np.std(concAr)/np.std(concAl)
      print "s%f \n"%(stddevs[i,j])
      #break 
      # hijacking Ts to store last concentrations
      ts = [concAl,concAr]
    
 
    
  writepickle(pickleName,ts,stddevs)    

  plt.pcolormesh(Deffs,dists,stddevs,cmap="Greys_r")
  plt.xlabel("dists")
  plt.ylabel("Deffs") 
  plt.gcf().savefig("test7.png",dpi=300) 

def test8():
  ## test block of right channel 
  params = Params()
  params.D12=0; params.D13=1000
  params.goodwinReaction = True
  params.meshName = "smallcube"
  result = Problem(params=params)


def figA(steps = 200, dt = 0.01, \
         DAb=1e9, DBb=1e9, DCb=1e9,\
         barrierLength = 100 * nm_to_um, 
         paraview=False,
         doplot=1):
  params = Params()
  params.paraview = paraview
  print paraview
  
  print "WRNING: this is a hack, since not updated correctly"
  params.volumeDom1 = np.prod(params.meshDim)
  print params.volumeDom1 
  yz = np.sqrt(params.volumeDom1/barrierLength)
  params.meshDim=np.array([barrierLength,yz,yz])
  print np.prod(params.meshDim)
  vols = np.array([params.volumeDom1 , params.volume_scalar2 , params.volume_scalar3])
  totVols = np.sum(vols)  
  volFracs = vols/totVols
  
  
  params.p=12.
  params.D12=1e9; params.D13=1e9; 
  params.DAb=DAb  
  params.DBb=DBb  
  params.DCb=DCb  
  cA=2; cB=1.5; cC=1.;
  params.cAbinit=cA; params.cAlinit=cA; params.cArinit=cA
  params.cBbinit=cB; params.cBlinit=cB; params.cBrinit=cB
  params.cCbinit=cC; params.cClinit=cC; params.cCrinit=cC
  params.steps = steps
  params.dt = dt 
  
  
  
  # exact w small enough time-step 
  kodes= np.array([5,1,1,1])
  vs   = np.array([5,1,0,0]) # reaction 1 occurs in compart2 (rxn 2 not)
  ks   = np.array([0,0,1,1]) # reaction 2 occurs in compart3 (rxn 1 not)
  
  # Only works if reaction happens -only- in one of the compartments (otherwise double counting)
  s2= 1/volFracs[1]
  s3= 1/volFracs[2]
  params.v0 = s2*vs[0]
  params.k0 = s3*ks[0]
  params.v1 = s2*vs[1]
  params.k1 = s3*ks[1]
  params.v2 = s2*vs[2]
  params.k2 = s3*ks[2]
  params.v3 = s2*vs[3]
  params.k3 = s3*ks[3]
  
  tode = scipy.linspace(0.,params.steps*params.dt,params.steps)
  y0ode=[params.cAlinit,params.cBlinit,params.cClinit]
  
  params.goodwinReaction = "opsplit2"
  params.meshName = "smallcube"
  results = Problem(params=params) 
  results.volFracs = volFracs 
  results.params = params 

  # for compare 
  import goodwin
  ks = np.concatenate(([1,1/params.Km,params.p],kodes))
  yode=goodwin.rxn(tode,y0ode,ks)
      
  if doplot:
    # plot first set 
    plotconcs1(results.ts,results.concs)
    plotconcs2(results.ts,results.concs)
    plotconcssum(results.ts,results.concs)
  
    plotODE(tode,yode)
    plt.gcf().savefig("testin.png",dpi=300)
  
    plt.figure()
    plotODE(tode,yode)

#print yode
  return results, tode, yode


  
    

  



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
    if(arg=="-test7buff"): 
      test7(buff=True)#sys.argv[i+1])
      quit()
    if(arg=="-test8"): 
      test8()#sys.argv[i+1])
      quit()
    if(arg=="-figA"): 
      figA()#sys.argv[i+1])
      quit()
    if(arg=="-genFreq"): 
      genFreqShifts()
      quit()
  





  raise RuntimeError("Arguments not understood")




