import matplotlib.pylab as plt
import scipy.integrate



a1=5;
kappa1=1.;
k1=1;
b1=1.;
alpha1=1;
beta1=1.; #0.6;
gamma1=1;
delta1=1; #0.8;
n=12.

def goodwinmodel(t,y0,a,title=""): 
  def f(y,t,a):
        X=y[0]
        Y=y[1]
        Z=y[2]
 

        #dt =[a1/(kappa1+k1*Z**n) - b1*X,  alpha1*X - beta1*Y, gamma1*Y - delta1*Z]
        #dt =[a1/(kappa1+k1*Z**n),0,0]
        dt =[a1/(kappa1+k1*Z**n)-b1*X,alpha1*X - beta1*Y,gamma1*Y- delta1*Z]
        return dt

  y = scipy.integrate.odeint(f,y0,t,args=(a,))
  return y 


def doit():
  t = scipy.linspace(0.,250.,100)
  #t = scipy.linspace(0.,50.,25)
  y0=[0,0,1]
  ks = [0]
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




