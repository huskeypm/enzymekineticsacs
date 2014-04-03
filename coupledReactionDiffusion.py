# 


# TODO 
# use Johan'vA2 meshes
# add in Johan'vA2 fluxes
# add back reactions 
# add right hand compart (remember, need to have more generall way of defining weak forms for all these rxns 


from dolfin import *
import numpy as np

## My reaction system 
# dc/dt = 
# Volume '1' is our diffusional domain
# Volume '2' is our ode domain 

debug = True 


## Params 
# compartments 
dist = Constant(1.)  # PKH what is this - dist between 1/2 and 1/3 

# kinetics 
kp = 1.0     
km = 0.6 
bT = 70.0   # [uM]  
Kd = km/kp; # [uM] 

# concentrations 
nComp = 6


# time steps 
if debug:
  steps=5
else:
  steps = 500

class Params():
  dt = 0.25  # [s] 
  D1   = 1.  # [um^2/s] Diff const within PDE (domain 1) 
  D12  = 1.  # [um^2/s] Diff const between domain 1 and 2
  D13  = 1.  # [um^2/s] Diff const between domain 1 and 3

  cA1init = 0.5 # [uM]
  cB1init = 0.5 # [uM]
  cA2init =1.0
  cB2init =1.0
  cA3init =0.1
  cB3init =0.1

  periodicSource=False # periodic source on CA2
  a = 0.3
  b = 0.2 



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


def Problem(params = Params()): 
  dt = Constant(params.dt)
  D1 = Constant(params.D1)   # diff within domain 1 
  D12 = Constant(params.D12) # diffusionb etween domain 1 and 2 
  D13 = Constant(params.D13)

  # Define mesh and function space 
  #mesh = UnitSquare(16,16)
  marker12 = 10 # boundary marker for domains 1->2
  marker13 = 11 # boundary marker for domains 1->3
  mesh = Mesh("cube.xml.gz")
  face_markers = MeshFunction("uint",mesh, "cube_face_markers.xml.gz")
  ds = Measure("ds")[face_markers]
      
  
  
  ##
  volumeDom1 = Constant(assemble(Constant(1.0)*dx,mesh=mesh))
  area = Constant(assemble(Constant(1.0)*ds(marker12),mesh=mesh))
  volume_scalar2 = Constant(4.*float(volumeDom1))
  volume_frac12 = volumeDom1/volume_scalar2 
  volume_scalar3 = Constant(4.*float(volumeDom1))
  volume_frac13 = volumeDom1/volume_scalar3
  
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
  expr = Expression("a*sin(b*t)",a=params.a,b=params.b,t=0)
  if params.periodicSource:
    print "Adding periodic source" 
    RHSA2 += expr*cA2_n*vA2*dx
  
  
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
  
  # Step in time
  # Step in time
  t   = 0.0
  T = steps*float(dt)
  tots=np.zeros([steps,nComp])
  concs=np.zeros([steps,nComp])
  ts=np.zeros(steps)
  j=0
  
  while (t   < T):
      solver.solve(problem, u_n.vector())
  
      ## store 
      # check values
      for i,ele in enumerate(split(u_n)):
        tot = assemble(ele*dx,mesh=mesh)
        vol = assemble(Constant(1.)*dx,mesh=mesh)
        conc = tot/vol  
        concs[j,i] = conc
        #print "Conc(%d) %f " % (i,conc)          
  
      tots[j,0] = assemble(cA1_n*dx) # PDE domain 
      tots[j,1] = assemble(cB1_n*dx)
      tots[j,2] = assemble(cA2_n/volume_frac12*dx) # scalar domain
      tots[j,3] = assemble(cB2_n/volume_frac12*dx)
      tots[j,4] = assemble(cA3_n/volume_frac13*dx)
      tots[j,5] = assemble(cB3_n/volume_frac13*dx)
      ts[j] = t    
      
      file << (u_n.split()[0], t)    

      ## update 
      #file << (u,vB2)
      j+=1
      t   += float(dt)
      expr.t = t 
      u0.vector()[:] = u_n.vector()
  
  return ts,concs,tots

def test12():
  ## test block of left channel 
  params = Params()
  params.D12=1; params.D13=0. 
  ts,concs,tots = Problem(params=params)

  ## assert 
  # for 5 timesteps 
  finalRef = np.array([  37.089814176,  37.089814176, 250.910186824, 250.910186824,25.6,25.6])
  final = tots[-1,:] 
  print final 
  for i in np.arange(nComp): 
    assert(np.abs(final[i] - finalRef[i])< 0.001), "Failed for species %d [%f/%f]" % (i,final[i],finalRef[i])

  return tots

def test13():
  ## test block of left channel 
  params = Params()
  params.D12=0; params.D13=1. 
  ts,concs,tots = Problem(params=params)

  ## assert 
  # for 5 timesteps 
  finalRef = np.array([  27.98166065 , 27.98166065 ,256.  ,       256.   ,       29.61833935,   29.61833935])
  final = tots[-1,:] 
  print final 
  for i in np.arange(nComp): 
    assert(np.abs(final[i] - finalRef[i])< 0.001), "Failed for species %d [%f/%f]" % (i,final[i],finalRef[i])

  return tots


def validation(): 
  test12()
  test13()




#
#validation()
