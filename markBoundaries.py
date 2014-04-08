#
# Kekenes-Huskey
# 

# for creating a box mesh that we can use with FEniCS simulations 

from dolfin import *
import numpy as np 

# outer
#cubeDim=([4,4,4]) 
cubeDim=([1,1,1]) 
class LeftBoundary(SubDomain): 
  def inside(self,x,on_boundary):
    return x[0] < 0+DOLFIN_EPS and on_boundary 

# outer
class RightBoundary(SubDomain): 
  def inside(self,x,on_boundary):
    return x[0] >= cubeDim[0]-DOLFIN_EPS and on_boundary 


def cube(mode="-cube"):
  # define mesh, function space
  mesh = UnitCube(8,8,8)
  V = FunctionSpace(mesh,"CG",1)
  mesh.coordinates()[:] = mesh.coordinates()[:] * cubeDim

  filePrefix = "cube"

  # define subdomains 
  subdomains = MeshFunction("uint",mesh,2)
  markers = []
  #elif(mode=="-dual"): 
  if 1: 
    leftBoundary = LeftBoundary()
    rightBoundary = RightBoundary()
    marker = 10
    leftBoundary.mark(subdomains,marker)
    markers.append(marker) 
    marker2 = 11
    rightBoundary.mark(subdomains,marker2)
    markers.append(marker2) 


  # cells 
  from dolfin import CellFunction
  cells = CellFunction("uint", mesh)
  cells.set_all(1) # not sure if this is right 

  # write 
  File(filePrefix+".xml.gz") << mesh 
  File(filePrefix+"_face_markers.xml.gz") << subdomains
  File(filePrefix+"_tetrahedron_attributes.xml.gz") << cells


  # report 
  V = FunctionSpace(mesh,"CG",1)
  ds = Measure("ds")[subdomains]
  totArea = 0
  for i,marker in enumerate(markers): 
    area = assemble(Constant(1.0)*ds(marker),mesh=mesh)       
    totArea+=area
    print "Marker: %d Area = %f nm^2 " % (marker,area) 
  vol  = assemble(Constant(1.0)*dx, mesh=mesh)
  print "Tot Area = %f nm^2 Vol = %f nm^3" % (assemble(Constant(1.0)*dx,mesh=mesh),vol) 
  

import sys
#
# Revisions
#       10.08.10 inception
#

if __name__ == "__main__":
  import sys
  scriptName= sys.argv[0]
  msg="""
Purpose: 
  For marking and writing a Dolfin mesh primitive 
  that is compatible with subcell
 
Usage:
"""
  msg+="  %s -cube/-nuo" % (scriptName)
  msg+="""
  
Notes:
  -cube simple cube with entire outer boundary marked 
  -dual simple cube with regions within/outside sphere marked
  -nuo simple example for nuo

"""
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  fileIn= sys.argv[1]
  if(len(sys.argv)==3):
    print "arg"

#  for i,arg in enumerate(sys.argv):
#    if(arg=="-cube" or arg=="-dual"):
#      cube(mode=arg)
#    if(arg=="-nuo"):
#      nuo()
  cube()




