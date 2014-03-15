FILENAME=main

TEX=pdflatex --shell-escape
BIBTEX=bibtex
#DVIPS=dvips -o $(FILENAME).ps
# CLEAN=${HOME}/bin/clean -t
INDEX=makeindex
INDEX_ARG=-s $(FILENAME).ist -t $(FILENAME).glg -o $(FILENAME).gls \
	$(FILENAME).glo
SINDEX=makesymbolindex -ih
RM=/bin/rm -f
GREP=grep
EGREP=egrep

alldirs=chapter_* appendix_*

default: pdf

all: pdf clean

nolink: pdf_nolink clean

tex::
	$(TEX) $(FILENAME).tex

tex_nolink::
	$(TEX) $(FILENAME)_nolink.tex

bib::
	# if ($(GREP) "\citation" $(FILENAME).aux > /tmp/make$$$$); then \
	$(BIBTEX) $(FILENAME);
	# fi

bib_nolink::
	if ($(GREP) "\citation" $(FILENAME)_nolink.aux > /tmp/make$$$$); then \
	      $(BIBTEX) $(FILENAME)_nolink; \
	fi

index::
	if (test -f $(FILENAME).idx); then \
	  # $(INDEX) $(FILENAME).idx; \
	  $(INDEX) $(INDEX_ARG); \
	fi

index_nolink::
	if (test -f $(FILENAME)_nolink.idx); then \
	      $(INDEX) $(FILENAME)_nolink.idx; \
	    fi

aindex::
	if (test -f $(FILENAME).adx); then \
	  $(INDEX) -o $(FILENAME).and -t $(FILENAME).alg $(FILENAME).adx; \
	fi

aindex_nolink::
	if (test -f $(FILENAME)_nolink.adx); then \
	      $(INDEX) -o $(FILENAME)_nolink.and -t $(FILENAME)_nolink.alg $(FILENAME)_nolink.adx; \
	    fi


los::
	if (test -f $(FILENAME).sdx); then \
	    $(SINDEX) $(FILENAME).sdx; \
	fi

los_nolink::
	if (test -f $(FILENAME)_nolink.sdx); then \
	        $(SINDEX) $(FILENAME)_nolink.sdx; \
	    fi


pdf::
	make tex
	make index
	make bib
	make tex
	make index
	make tex
	make tex

pdf_nolink::
	make tex_nolink
	make bib_nolink
	make tex_nolink
	make tex_nolink
	make tex_nolink
	make index_nolink
	make aindex_nolink
	make los_nolink
	if (test -f $(FILENAME)_nolink.idx || test -f $(FILENAME)_nolink.adx || \
	  test -f $(FILENAME)_nolink.sdx); then \
	      make tex; \
	      make tex; \
	      make index; \
	      make aindex; \
	      make los; \
	      make tex; \
	      make tex; \
	    fi

abstract::
	$(TEX) abstract_$(FILENAME).tex

ps::
	$(DVIPS) $(FILENAME).dvi

myclean:
	rm -rf main.log
	#rm -rf main.pdf
	rm -rf main.glo
	rm -rf main.lof
	rm -rf main.lot
	rm -rf main.out
	rm -rf main.toc
	rm -rf main.aux
	rm -rf main.brf
	rm -rf main.ist
	rm -rf main.idx
	rm -rf main.glg
	rm -rf main.gls
	rm -rf main.bbl
	rm -rf main.blg


clean::
	$(CLEAN)

ps_clean: clean
	$(RM) $(FILENAME).ps

very_clean: ps_clean
	for name in $(alldirs); \
	do \
	 (cd $${name}; make ps_clean); \
	done

ispell_clean::
	$(RM) *.tex.bak; \
	for name in $(alldirs); \
	do \
	  (cd $${name}; $(RM) *.tex.bak); \
	done

update: dvi
	for name in $(alldirs); \
	do \
	 (echo "cd $${name}"; \
	 cd $${name}; make); \
	done
