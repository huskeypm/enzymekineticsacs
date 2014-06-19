import matplotlib.pylab as plt
import scipy.integrate



kappa1=1.;
ikm=1;
p=12.
k0=5;
k1=1;
k2=1;
k3=1; #0.8;

# single compartment 
def rxn(t,y0,a,title=""): 
  def f(y,t,a):
    X,Y,Z = y
    kappa1,ikm,p,k0,k1,k2,k3 = a 

    dt =[k0/(kappa1+(ikm*Z)**p)-k1*X,\
      k1*X - k2*Y,\
      k2*Y- k3*Z]
    return dt

  y = scipy.integrate.odeint(f,y0,t,args=(a,))
  return y 
#
## left compartment 
#def goodwinmodelComp1(t,y0,a,title=""): 
#  def f(y,t,a):
#    X,Y,Z = y
#    kappa1,ikm,p,k0,k1,k2,k3 = a 
#
#    dt =[k0/(kappa1+ikm*Z**p)-k1*X,\
#      k1*X ,\
#      0] # no reaction      
#    return dt
#
#  y = scipy.integrate.odeint(f,y0,t,args=(a,))
#  return y 
#
## left compartment 
#def goodwinmodelComp3(t,y0,a,title=""): 
#  def f(y,t,a):
#    X,Y,Z = y
#    kappa1,ikm,p,k0,k1,k2,k3 = a 
#
#    dt =[0, # no reaction
#      0 - k2*Y,\
#      k2*Y- k3*Z]
#    return dt
#
#  y = scipy.integrate.odeint(f,y0,t,args=(a,))
#  return y 
#

def doit():
  t = scipy.linspace(0.,100.,1000)
  #t = scipy.linspace(0.,50.,25)
  y0=[0,0,1]
  ks = [kappa1,ikm,p,k0,k1,k2,k3]
  y=goodwinmodel(t,y0,ks)
  
  plt.plot(t,y[:,0],"r-",label="x")
  plt.plot(t,y[:,1],"b-",label="y")
  plt.plot(t,y[:,2],"g-",label="z")    
  plt.ylabel("$[y_i]$")
  plt.xlabel("t")
  plt.legend(loc=0)
  plt.gcf().savefig("test.png",dpi=300)
                   

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

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  fileIn= sys.argv[1]
  if(len(sys.argv)==3):
    print "arg"

  for i,arg in enumerate(sys.argv):
    if(arg=="-validation"):
#      arg1=sys.argv[i+1] 
      doit()
      quit()
  





  raise RuntimeError("Arguments not understood")




