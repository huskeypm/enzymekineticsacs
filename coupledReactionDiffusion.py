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


## Params 
field=False
if(field==True):
  Dii = Constant((1.0,1.0,1.0))
  Dii = Constant((5.,0.1,0.1))
  Dij = diag(Dii)
else:
  Dij = Constant(1.0)

D = Dij
dist = Constant(1.)  # PKH what is this 
## params 
kp = 1.0     
km = 0.6 
cA1init = 0.5 # [uM]
cB1init = 0.5 # [uM]
cA2init =1.0
cB2init =1.0

bT = 70.0   # [uM]  
Kd = km/kp; # [uM] 
dt     = 1.0e-03  
dt = Constant(0.25)



class InitialConditions(Expression):

  def eval(self, values, x):
    # edge  
    #if (x[0] < 0.5 and np.linalg.norm(x[1:2]) < 0.5):
    # corner 
    #oif (np.linalg.norm(x -np.zeros(3) ) < 0.5):
    if 1:
      values[0] = cA1init
      values[1] = cB1init
      values[2] = cA2init
      values[3] = cB2init
    #else:
    #  values[0] = 0         
    #  values[1] = 0
    #  values[2] = 0
    #  values[3] = 0
  def value_shape(self):
    return (4,)

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



# Define mesh and function space 
#mesh = UnitSquare(16,16)
simple=False
if(simple):
  xMax= 2
  yMax = 2 * 0.59
  zMax = 5 
  mesh = UnitCube(8,8,8)      
  mesh.coordinates()[:]= np.array([xMax,yMax,zMax])* mesh.coordinates()
else:
  mesh = Mesh("channel-mesh-0.xml.gz")
  face_markers = MeshFunction("uint",mesh, "channel-mesh-0_face_markers.xml.gz")
  ds = Measure("ds")[face_markers]

##
volume = Constant(assemble(Constant(1.0)*dx,mesh=mesh))
area = Constant(assemble(Constant(1.0)*ds(10),mesh=mesh))
volume_scalar = Constant(4.*float(volume))
volume_frac = volume/volume_scalar 

# functions 
V = FunctionSpace(mesh,"CG",1)
R = FunctionSpace(mesh,"R",0)
ME = V *V*R
ME = MixedFunctionSpace([V,V,R,R])

# Trial and Test functions 
du = TrialFunction(ME) 
vA1,vB1,vA2,vB2  = TestFunctions(ME)

# Define function
u_n = Function(ME) # current soln
u0 = Function(ME) # prev soln

# split mixed functions
cA1_n,cB1_n,cA2_n,cB2_n = split(u_n)
cA1_0,cB1_0,cA2_0,cB2_0 = split(u0)

# Init conts
#init_cond = InitialConditions()
init_cond = InitialConditions()
u_n.interpolate(init_cond)
u0.interpolate(init_cond)


## Weak forms for RHS  
# See notetaker 121213 notes for details 

# Diffusion
#if(field==False):
#  RHSA1 = -Dij*inner(grad(c),grad(vA1))*dx  
#  RHSB1 = -Dij*inner(grad(cb),grad(vB1))*dx 
#else:
RHSA1 = -inner(Dij*grad(cA1_n),grad(vA1))*dx  
RHSB1 = -inner(Dij*grad(cB1_n),grad(vB1))*dx 
RHSA2 = Constant(0)*vA2*dx # for consistency
RHSB2 = Constant(0)*vB2*dx

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


# Time derivative and diffusion of field species
# (dc/dt) = RHS  --> c1-cA1_0 - dt * RHS = 0
# WRONG LA1 = (c-cA1_0)c*vA1*dx - cA1_0*vA1*dx - dt * RHSA1
#LB1 = cb*vB1*dx - cB1_0*vB1*dx - dt * RHSB1
#LA2 = cs*vA2*dx - cA2_0*vA2*dx - dt * RHSA2
#LB2 = ct*vB2*dx - cB2_0*vB2*dx - dt * RHSB2
LA1 = (cA1_n-cA1_0)*vA1/dt*dx - RHSA1 
LB1 = (cB1_n-cB1_0)*vB1/dt*dx - RHSB1 

# Flux to mesh domain
LA1 -= Dij*(cA2_n-cA1_n)*vA1/dist*ds(10)
LB1 -= Dij*(cB2_n-cB1_n)*vB1/dist*ds(10)


# Time derivative of scalar species and flux from scalar domain 
LA2 = (cA2_n-cA2_0)*vA2/(dt*volume_frac)*dx - RHSA2
LB2 = (cB2_n-cB2_0)*vB2/(dt*volume_frac)*dx - RHSB2
LA2 += Dij*(cA2_n-cA1_n)*vA2/dist*ds(10)
LB2 += Dij*(cB2_n-cB1_n)*vB2/dist*ds(10)

# compbine
L = LA1 + LB1 + LA2 + LB2

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
steps=5
T = steps*float(dt)
nComp = 4
tots=np.zeros([steps,nComp])
ts=np.zeros(steps)
j=0

while (t   < T):
    solver.solve(problem, u_n.vector())

    # check values
    #for i,ele in enumerate(split(u_n)):
    #  tot = assemble(ele*dx,mesh=mesh)
    #  vol = assemble(Constant(1.)*dx,mesh=mesh)
    #  conc = tot/vol  
    #  tots[j,i]=tot  
    #  print "Conc(%d) %f " % (i,tot/vol)
    tots[j,0] = assemble(cA1_n*dx)
    tots[j,1] = assemble(cB1_n*dx)
    tots[j,2] = assemble(cA2_n/volume_frac*dx)
    tots[j,3] = assemble(cB2_n/volume_frac*dx)
    ts[j] = t    
    
    file << (u_n.split()[0], t)    
    #file << (u,vB2)
    j+=1
    t   += float(dt)
    u0.vector()[:] = u_n.vector()



## assert 
final = tots[-1,:] 
finalRef = np.array([  510.33298176,  510.33298176, 3989.66701824, 3989.66701824])
for i in np.arange(nComp): 
  assert(np.abs(final[i] - finalRef[i])< 0.001), "Failed for species %d [%f/%f]" % (i,final[i],finalRef[i])


#
