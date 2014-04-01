unalias python
unalias ipython
alias ipn="ipython notebook --pylab inline"
export MYPATH=/net//home/huskeypm/sources/mypython/
export PYTHONPATH=$PYTHONPATH:$MYPATH/lib/python2.7/site-packages/

source /net/home/huskeypm/Sources/share/dolfin/dolfin.conf
export BOOST_DIR=/usr
export PYTHONPATH=/net/home/huskeypm/Sources/smolfin/:$PYTHONPATH
export PYTHONPATH=/net/home/huskeypm/Sources/smolfin/:$PYTHONPATH
export PYTHONPATH=/net/home/huskeypm/Sources/homogenization/:$PYTHONPATH



export SRCDIR=/net/home/huskeypm/Sources/
export VER=2.7
export PREFIX=$SRCDIR/gamer-src/
export PYTHONPATH=$PYTHONPATH:$PREFIX/lib/python$VER/site-packages
