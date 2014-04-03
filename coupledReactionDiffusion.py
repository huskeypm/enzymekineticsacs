# 
# This code performs a reaction-diffusion simulation within a cube 
# General reaction-diffusion example for buffered Ca2+ 
# Largely borrowed from Cahn-Hilliard example 
#
# Validation
# -Buffering seems to be correct (set cInit to 45 everywhere, checked that cb final agreed with ExcessBuffer output)
# -anistripic diffusion is at least qualitiatively correct

from dolfin import *
import numpy as np

## My reaction system 
# dc/dt = del D del c - R(c,cB)
# dcB/dt = del D del cB + R(c,cB)
# R(c,CB) = kp * (B - cB)*c - km*cB

## Params 
field=False
if(field==True):
  Dii = Constant((1.0,1.0,1.0))
  Dii = Constant((5.,0.1,0.1))
  Dij = diag(Dii)
else:
  Dij = Constant(1.0)

kp = 1.0     
km = 0.6 
cInit = 5.0 # [uM]
bT = 70.0   # [uM]  
Kd = km/kp; # [uM] 
dt     = 1.0e-03  


xMax= 2
yMax = 2 * 0.59
zMax = 5 

class InitialConditions(Expression):

  def eval(self, values, x):
    # edge  
    #if (x[0] < 0.5 and np.linalg.norm(x[1:2]) < 0.5):
    # corner 
    if (np.linalg.norm(x -np.zeros(3) ) < 0.5):
      values[0] = cInit
      values[1] = 2*cInit
    else:
      values[0] = 0         
      values[1] = 0
  def value_shape(self):
    return (2,)

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
mesh = UnitCube(8,8,8)      
mesh.coordinates()[:]= np.array([xMax,yMax,zMax])* mesh.coordinates()
V = FunctionSpace(mesh,"CG",1)
ME = V *V 

# Trial and Test functions 
du = TrialFunction(ME) 
q,v  = TestFunctions(ME)

# Define function
u = Function(ME) # current soln
u0 = Function(ME) # prev soln

# split mixed functions
c,cb = split(u)
c0,cb0 = split(u0)

# Init conts
#init_cond = InitialConditions()
init_cond = InitialConditions()
u.interpolate(init_cond)
u0.interpolate(init_cond)


## Weak forms for RHS  
# See notetaker 121213 notes for details 

# Diffusion
#if(field==False):
#  RHS1 = -Dij*inner(grad(c),grad(q))*dx  
#  RHS2 = -Dij*inner(grad(cb),grad(v))*dx 
#else:
RHS1 = -inner(Dij*grad(c),grad(q))*dx  
RHS2 = -inner(Dij*grad(cb),grad(v))*dx 

# Reaction: b + c --kp--> cb,  
#           b + c <-km--- cb
R = np.array([
  [-kp,km],     # s
  [kp,-km]      # p
  ])


# operator splitting 
opSplit=False
if(opSplit==False):
  RHS1 += (R[0,0]*(bT-cb)*c*q + R[0,1]*cb*q)*dx
  RHS2 += (R[1,0]*(bT-cb)*c*v + R[1,1]*cb*v)*dx


# Add in time dependence 
# (dc/dt) = RHS  --> c1-c0 - dt * RHS = 0
L1 = c*q*dx - c0*q*dx - dt * RHS1 
L2 = cb*v*dx - cb0*v*dx - dt * RHS2 
L = L1 + L2

# Compute directional derivative about u in the direction of du (Jacobian)
# (for Newton iterations) 
a = derivative(L, u, du)


# Create nonlinear problem and Newton solver
problem = MyEqn(a, L)
solver = NewtonSolver("lu")
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6

# Output file
file = File("output.pvd", "compressed")

# Step in time
t = 0.0
T = 25*dt
while (t < T):
    t += dt
    u0.vector()[:] = u.vector()
    solver.solve(problem, u.vector())

    if(opSplit==True):
      # Check Johans 121213/4 email regarding using dof maps and sympy functions here 
      1
      
    #file << (u.split()[0], t)
    file << (u,t)

    # check values
    for i,ele in enumerate(split(u)):
      tot = assemble(ele*dx,mesh=mesh)
      vol = assemble(Constant(1.)*dx,mesh=mesh)
      print "Conc(%d) %f " % (i,tot/vol)




#
