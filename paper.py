Dwater = 2.6 # um2/ms (see paper) 
DATPbulk = 0.183 # um2/ms
DATP = 0.145 # um2/ms
dt = 0.03 # varies below []
from coupledReactionDiffusion3comp import *

def RunCases(mode="none"):
  print "SDS"; print mode 
  if mode=="O" or mode=="all":
    steps = 2 
    resultsFast, tode, yode = figA(steps =steps, dt = 0.03,doplot=0,\
       DAb=1e9,DBb=1e9,DCb=1e9,\
       pickleName="shit.pkl")
  if mode=="A" or mode=="all":
    steps = 501
    resultsFast, tode, yode = figA(steps =steps, dt = 0.03,doplot=0,\
       DAb=1e9,DBb=1e9,DCb=1e9,\
       pickleName="resultsFast.pkl")
#    resultsSlow, tode, yode = figA(steps =steps, dt = 0.03,doplot=0,\
#       DAb=1e2,DBb=1e2,DCb=1e2,\
#       pickleName="resultsSlow.pkl")

  if mode=="B" or mode=="all":
    steps = 501
    resultsWater, tode, yode = figA(\
       steps =steps, dt = 0.03,doplot=0,\
       DAb=Dwater,DBb=Dwater,DCb=Dwater,barrierLength=0.01,paraview=False,\
       pickleName="resultsWater.pkl")

     
  if mode=="C" or mode=="all":
    steps = 1501
    resultsWaterB100, tode, yode = figA(\
       steps =steps, dt = 0.05,doplot=0,\
       DAb=Dwater,DBb=Dwater,DCb=Dwater,barrierLength=0.1,paraview=False,\
       pickleName="resultsWaterB100.pkl")

  if mode=="D" or mode=="all":
    resultsCslow, tode, yode = figA(\
                                    #steps =501, dt = 0.03,doplot=0,DAb=1e3,DBb=1e2,DCb=1e3)
      steps =701, dt = 0.03,doplot=0,\
      DAb=Dwater,DBb=Dwater,DCb=DATP,barrierLength=0.01,paraview=False,\
      pickleName="resultsCslow.pkl")

  if 1:
    steps = 501 
    steps = 201
    dt = 0.03 # osc
    #dt = 0.01
    #Dref = 1e2
    dt = 0.03; scale = 0.5 # osc
    dt = 0.03; scale = 0.2; # osc 
    #steps = 1501; dt = 0.01; scale = 0.1 # noosc
    Dref = DATP # [um^2/ms]
    steps = 1501

    scales = [0.1,1.0,10] # kills osc for 0.1
    scales = [0.5,1.0,2.] # osc 
    scales = [0.2,1.0,5.] # osc

  if mode=="E" or mode=="all":
    resultsABslowCveryslow, tode, yode = figA(steps =steps, dt =dt,doplot=0,\
                                              barrierLength=0.01,paraview=False,
                                              DAb=Dref,DBb=Dref,DCb=Dref*scales[0],\
                                              pickleName="resultsABslowCveryslow.pkl")   
  if mode=="F" or mode=="all":
    resultsAllslow, tode, yode = figA(steps =steps, dt = dt,doplot=0,\
                                              barrierLength=0.01,paraview=False,
                                              DAb=Dref,DBb=Dref,DCb=Dref*scales[1],\
                                              pickleName="resultsAllslow.pkl")   
  if mode=="G" or mode=="all":
    resultsABslowCfast, tode, yode = figA(steps =steps, dt = dt,doplot=0,\
                                              barrierLength=0.01,paraview=False,
                                              DAb=Dref,DBb=Dref,DCb=Dref*scales[2],\
                                              pickleName="resultsABslowCfast.pkl")   




#resultsCslow
#resultsABslowCveryslow
#resultsAllslow
#resultsABslowCfast


#!/usr/bin/env python
import sys
##################################
#
# Revisions
#       10.08.10 inception
#
##################################


#
# Message printed when program run without arguments 
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

#
# MAIN routine executed when launching this script from command line 
#
if __name__ == "__main__":
  import sys
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-run"):
      arg1=sys.argv[i+1] 
      RunCases(mode=arg1)
  





  #raise RuntimeError("Arguments not understood")




