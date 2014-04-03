# 


# TODO 
# use Johan's meshes
# add in Johan's fluxes
# add back reactions 
# add right hand compart (remember, need to have more generall way of defining weak forms for all these rxns 


from dolfin import *
import numpy as np

## My reaction system 
# dc/dt = 


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
cInit = 0.5 # [uM]
cbInit = 0.5 # [uM]
sInit =1.0
tInit =1.0

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
      values[0] = cInit
      values[1] = cInit
      values[2] = sInit
      values[3] = sInit
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
q,v,s,t  = TestFunctions(ME)

# Define function
u_n = Function(ME) # current soln
u0 = Function(ME) # prev soln

# split mixed functions
c_n,cb_n,cs_n,ct_n = split(u_n)
c0,cb0,cs0,ct0 = split(u0)

# Init conts
#init_cond = InitialConditions()
init_cond = InitialConditions()
u_n.interpolate(init_cond)
u0.interpolate(init_cond)


## Weak forms for RHS  
# See notetaker 121213 notes for details 

# Diffusion
#if(field==False):
#  RHS1 = -Dij*inner(grad(c),grad(q))*dx  
#  RHS2 = -Dij*inner(grad(cb),grad(v))*dx 
#else:
RHS1 = -inner(Dij*grad(c_n),grad(q))*dx  
RHS2 = -inner(Dij*grad(cb_n),grad(v))*dx 
RHS3 = Constant(0)*s*dx # for consistency
RHS4 = Constant(0)*t*dx

# Reaction: b + c --kp--> cb,  
#           b + c <-km--- cb
R = np.array([
  [-kp,km],     # s
  [kp,-km]      # p
  ])


# operator splitting 
rxn=False
#opSplit=True # just turning off reaction s
if(rxn):
  # no rxn in PDE part   
  #RHS1 += (R[0,0]*(bT-cb_n)*c_n*q + R[0,1]*cb_n*q)*dx
  #RHS2 += (R[1,0]*(bT-cb_n)*c_n*v + R[1,1]*cb_n*v)*dx

  RHS3 += (R[0,0]*(bT-ct_n)*cs_n*s + R[0,1]*ct_n*s)*dx
  RHS4 += (R[1,0]*(bT-ct_n)*cs_n*t + R[1,1]*ct_n*t)*dx


# Time derivative and diffusion of field species
# (dc/dt) = RHS  --> c1-c0 - dt * RHS = 0
# WRONG L1 = (c-c0)c*q*dx - c0*q*dx - dt * RHS1
#L2 = cb*v*dx - cb0*v*dx - dt * RHS2
#L3 = cs*s*dx - cs0*s*dx - dt * RHS3
#L4 = ct*t*dx - ct0*t*dx - dt * RHS4
L1 = (c_n-c0)*q/dt*dx - RHS1 
L2 = (cb_n-cb0)*v/dt*dx - RHS2 

# Flux to mesh domain
L1 -= Dij*(cs_n-c_n)*q/dist*ds(10)
L2 -= Dij*(ct_n-cb_n)*v/dist*ds(10)


# Time derivative of scalar species and flux from scalar domain 
L3 = (cs_n-cs0)*s/(dt*volume_frac)*dx - RHS3
L4 = (ct_n-ct0)*t/(dt*volume_frac)*dx - RHS4
L3 += Dij*(cs_n-c_n)*s/dist*ds(10)
L4 += Dij*(ct_n-cb_n)*t/dist*ds(10)

# compbine
L = L1 + L2 + L3 + L4

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
t = 0.0
steps=5
T = steps*float(dt)
nComp = 4
tots=np.zeros([steps,nComp])
ts=np.zeros(steps)
j=0

while (t < T):
    solver.solve(problem, u_n.vector())

    # check values
    #for i,ele in enumerate(split(u_n)):
    #  tot = assemble(ele*dx,mesh=mesh)
    #  vol = assemble(Constant(1.)*dx,mesh=mesh)
    #  conc = tot/vol  
    #  tots[j,i]=tot  
    #  print "Conc(%d) %f " % (i,tot/vol)
    tots[j,0] = assemble(c_n*dx)
    tots[j,1] = assemble(cb_n*dx)
    tots[j,2] = assemble(cs_n/volume_frac*dx)
    tots[j,3] = assemble(ct_n/volume_frac*dx)
    ts[j] = t
    
    file << (u_n.split()[0], t)
    #file << (u,t)
    j+=1
    t += float(dt)
    u0.vector()[:] = u_n.vector()



## assert 
final = tots[-1,:] 
finalRef = np.array([  510.33298176,  510.33298176, 3989.66701824, 3989.66701824])
for i in np.arange(nComp): 
  assert(np.abs(final[i] - finalRef[i])< 0.001), "Failed for species %d [%f/%f]" % (i,final[i],finalRef[i])


#
