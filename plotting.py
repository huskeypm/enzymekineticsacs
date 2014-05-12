### For plotting figuyres for manuscruipt
from coupledReactionDiffusion3comp import *
import numpy as np 

idxA1 = 0 # PDE 
idxB1 = 1
idxC1 = 2
idxA2 = 3 # L compartment 
idxB2 = 4
idxC2 = 5
idxA3 = 6 # R compartment 
idxB3 = 7
idxC3 = 8



def plotconcs(ts,concs):
  plt.figure()
  plt.subplot(311)
  c = concs[:,idxA2]
  #plt.plot(ts,c/c[0],'r-',label="A2")
  plt.plot(ts,c,'r-',label="A2")  
  c = concs[:,idxA1]
  #plt.plot(ts,c/c[0],'r.',label="A1")
  plt.plot(ts,c,'r.',label="A1")
  c = concs[:,idxA3]
  #plt.plot(ts,c/c[0],'r--',label="A3")
  plt.plot(ts,c,'r--',label="A3")
  plt.xlabel("t") 
  plt.ylabel("C/C[0]")
  plt.legend(loc=2)
    
  plt.subplot(312)
  c = concs[:,idxB2]
  #plt.plot(ts,c/c[0],'b-',label="B2")
  plt.plot(ts,c,'b-',label="B2")
  c = concs[:,idxB1]
  #plt.plot(ts,c/c[0],'b-',label="B2")
  plt.plot(ts,c,'b-',label="B2")
  c = concs[:,idxB3]
  #plt.plot(ts,c/c[0],'b--',label="B3")
  plt.plot(ts,c,'b--',label="B3")
  plt.xlabel("t") 
  plt.ylabel("C/C[0]")
  plt.legend(loc=2)    
    
  plt.subplot(313)
  c = concs[:,idxC2]
  #plt.plot(ts,c/c[0],'g-',label="C2")
  plt.plot(ts,c,'g-',label="C2")
  c = concs[:,idxC1]
  #plt.plot(ts,c/c[0],'g.',label="C1")
  plt.plot(ts,c,'g.',label="C1")
  c = concs[:,idxC3]
  #plt.plot(ts,c/c[0],'g--',label="C3")
  plt.plot(ts,c,'g--',label="C3")
  plt.xlabel("t") 
  plt.ylabel("C/C[0]")
  plt.legend(loc=2)        
   
    
def plotconcs1(ts,concs):
  plt.figure()
  c = concs[:,idxA2]
  plt.plot(ts,c,'r+',label="A2")  
  c = concs[:,idxB2]
  plt.plot(ts,c,'b+',label="B2")
  c = concs[:,idxC2]
  plt.plot(ts,c,'g+',label="C2")
  plt.ylabel("$[y_i]$ [uM]")
  plt.xlabel("t [ms]") 
  plt.legend(loc=0)     

def plotconcs2Barrier(ts,concs):
  #plt.figure()
  c = concs[:,idxA1]
  plt.plot(ts,c,'r.',label="A1")  
  c = concs[:,idxB1]
  plt.plot(ts,c,'b.',label="B1")
  c = concs[:,idxC1]
  plt.plot(ts,c,'g.',label="C1")
  plt.ylabel("$[y_i]$ [uM]")
  plt.xlabel("t [ms]") 
  plt.legend(loc=0)   
    
def plotconcs3(ts,concs):
  #plt.figure()
  c = concs[:,idxA3]
  plt.plot(ts,c,'rx',label="A3")  
  c = concs[:,idxB3]
  plt.plot(ts,c,'bx',label="B3")
  c = concs[:,idxC3]
  plt.plot(ts,c,'gx',label="C3")
  plt.ylabel("$[y_i]$ [uM]")
  plt.xlabel("t [ms]") 
  plt.legend(loc=0)   
    
def plotconcssum(ts,concs,volFracs=np.array([1,1,1])):
  #plt.figure()
    
  c = concs[:,idxA1]*volFracs[0]+concs[:,idxA2]*volFracs[1]+concs[:,idxA3]*volFracs[2]
  plt.plot(ts,c,'ro',label="At")  
  c = concs[:,idxB1]*volFracs[0]+concs[:,idxB2]*volFracs[1]+concs[:,idxB3]*volFracs[2]
  plt.plot(ts,c,'bo',label="Bt")
  c = concs[:,idxC1]*volFracs[0]+concs[:,idxC2]*volFracs[1]+concs[:,idxC3]*volFracs[2]
  plt.plot(ts,c,'go',label="Ct")
  plt.ylabel("$[y_i]$ [uM]")
  plt.xlabel("t [ms]") 
  plt.legend(loc=0)    
    
def plotODE(tode,yode):    
  plt.plot(tode,yode[:,0],"r-",label="xODE")
  plt.plot(tode,yode[:,1],"b-",label="yODE")
  plt.plot(tode,yode[:,2],"g-",label="zODE")
  plt.ylabel("$[y_i]$ [uM]")
  plt.xlabel("t [ms]") 
  plt.legend(loc=0)    
