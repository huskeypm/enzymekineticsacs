import numpy as np
import matplotlib.pylab as plt 

# plots FFT 
def doFFT(t,x,doplot=False):
  N = np.float(np.shape(x)[0]);
  
  Fs = np.float(np.shape(t)[0])
  xdft = np.fft.fft(x);
  xdft = xdft[1:N/2+1];
  #print xdft
  psdx = (1/(Fs*N))*np.abs(xdft)**2;
  psdx *=2   #[[2:end-1);
  dt = t[-1]-t[0]  
  freq = np.linspace(0,Fs/(2*dt),N/2.)
  if(doplot):  
    plt.plot(freq,np.log10(psdx)); #grid on;
    plt.title('Power spectral density');
    plt.xlabel('Frequency (Hz)'); plt.ylabel('Power/Frequency (dB/Hz)');
    
  return freq,psdx

def freq(yt,t,doplot=False,oldstuff=True):
  if oldstuff:
    tss = 100
    idx = np.where(t > tss)
    cA1 = np.ndarray.flatten(yt[idx,8])
    ts = t[idx]
  else:
    cA1 = yt
    ts = t 
    
  cA1m = cA1-np.mean(cA1)
  f,p = doFFT(ts,cA1m,doplot=doplot)   
  return f,p  

# validate
def doit():
  Fs = 1000; # frames per sec
  sec = 10   # sec
  t = np.linspace(0,sec-1/Fs,sec*Fs)
  v = 100. # Hz
  x = np.cos(2*pi*v*t)+1*0.01*np.random.randn(np.shape(t)[0]);
    
  doFFT(t,x,doplot=True)   

#print t    

