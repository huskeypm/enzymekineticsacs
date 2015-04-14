### For plotting figuyres for manuscruipt
import matplotlib.pylab as plt 
from coupledReactionDiffusion3comp import *
import numpy as np 

idxAb = 0 # PDE 
idxBb = 1
idxCb = 2
idxAl = 3 # L compartment 
idxBl = 4
idxCl = 5
idxAr = 6 # R compartment 
idxBr = 7
idxCr = 8



def plotconcs(ts,concs):
  plt.figure()
  plt.subplot(311)
  c = concs[:,idxAl]
  #plt.plot(ts,c/c[0],'r-',label="Al")
  plt.plot(ts,c,'r-',label="Al")  
  c = concs[:,idxAb]
  #plt.plot(ts,c/c[0],'r.',label="Ab")
  plt.plot(ts,c,'r.',label="Ab")
  c = concs[:,idxAr]
  #plt.plot(ts,c/c[0],'r--',label="Ar")
  plt.plot(ts,c,'r--',label="Ar")
  plt.xlabel("t") 
  plt.ylabel("C/C[0]")
  plt.legend(loc=2)
    
  plt.subplot(312)
  c = concs[:,idxBl]
  #plt.plot(ts,c/c[0],'b-',label="Bl")
  plt.plot(ts,c,'b-',label="Bl")
  c = concs[:,idxBb]
  #plt.plot(ts,c/c[0],'b-',label="Bl")
  plt.plot(ts,c,'b-',label="Bl")
  c = concs[:,idxBr]
  #plt.plot(ts,c/c[0],'b--',label="Br")
  plt.plot(ts,c,'b--',label="Br")
  plt.xlabel("t") 
  plt.ylabel("C/C[0]")
  plt.legend(loc=2)    
    
  plt.subplot(313)
  c = concs[:,idxCl]
  #plt.plot(ts,c/c[0],'g-',label="Cl")
  plt.plot(ts,c,'g-',label="Cl")
  c = concs[:,idxCb]
  #plt.plot(ts,c/c[0],'g.',label="Cb")
  plt.plot(ts,c,'g.',label="Cb")
  c = concs[:,idxCr]
  #plt.plot(ts,c/c[0],'g--',label="Cr")
  plt.plot(ts,c,'g--',label="Cr")
  plt.xlabel("t") 
  plt.ylabel("C/C[0]")
  plt.legend(loc=2)        
   
    
def plotconcs1(ts,concs):
  plt.figure()
  c = concs[:,idxAl]
  plt.plot(ts,c,'r+',label="Al")  
  c = concs[:,idxBl]
  plt.plot(ts,c,'b+',label="Bl")
  c = concs[:,idxCl]
  plt.plot(ts,c,'g+',label="Cl")
  plt.ylabel("$[y_i]$ [uM]")
  plt.xlabel("t [ms]") 
  plt.legend(loc=0)     

def plotconcs2Barrier(ts,concs):
  #plt.figure()
  c = concs[:,idxAb]
  plt.plot(ts,c,'r.',label="Ab")  
  c = concs[:,idxBb]
  plt.plot(ts,c,'b.',label="Bb")
  c = concs[:,idxCb]
  plt.plot(ts,c,'g.',label="Cb")
  plt.ylabel("$[y_i]$ [uM]")
  plt.xlabel("t [ms]") 
  plt.legend(loc=0)   
    
def plotconcs3(ts,concs):
  #plt.figure()
  c = concs[:,idxAr]
  plt.plot(ts,c,'rx',label="Ar")  
  c = concs[:,idxBr]
  plt.plot(ts,c,'bx',label="Br")
  c = concs[:,idxCr]
  plt.plot(ts,c,'gx',label="Cr")
  plt.ylabel("$[y_i]$ [uM]")
  plt.xlabel("t [ms]") 
  plt.legend(loc=0)   
    
def plotconcssum(ts,concs,volFracs=np.array([1,1,1])):
  #plt.figure()
    
  c = concs[:,idxAb]*volFracs[0]+concs[:,idxAl]*volFracs[1]+concs[:,idxAr]*volFracs[2]
  plt.plot(ts,c,'ro',label="At")  
  c = concs[:,idxBb]*volFracs[0]+concs[:,idxBl]*volFracs[1]+concs[:,idxBr]*volFracs[2]
  plt.plot(ts,c,'bo',label="Bt")
  c = concs[:,idxCb]*volFracs[0]+concs[:,idxCl]*volFracs[1]+concs[:,idxCr]*volFracs[2]
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
