import matplotlib.pylab as plt
import scipy.integrate



kf=1;
kb=1;
p=1

# left compartment 
def rxn(t,y0,a,title=""): 
  def f(y,t,a):
    X,Y,Z = y
    r,k1,s = a

    dt =[   r - k1*X,\
         k1*X - s*Y,\
         0] # no reaction      
    return dt

  y = scipy.integrate.odeint(f,y0,t,args=(a,))
  return y 


