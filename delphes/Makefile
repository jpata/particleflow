# S.Chekanov

# define here PYTHIA and HEPMC directories
ifndef PYTHIA8_DIR
$(error PYTHIA8_DIR env variable is not set. Run setup.sh first)
endif


ifndef PROMC
$(error PROMC env variable is not set. Run setup.sh first)
endif

include ${PROMC}/etc/config.mk
include ${ROOTSYS}/etc/Makefile.arch


# Root variables
ROOTCFLAGS    = $(shell root-config --nonew --cflags)
ROOTLIBS      = $(shell root-config --nonew --libs)
ROOTGTTLIBS   = $(shell root-config --nonew --glibs)
CXXFLAGS     += $(ROOTCFLAGS)

LIBDIRARCH=lib/
OutPutOpt     = -o
LIBS         += -L$(PROMC)/lib -lpromc -lprotoc -lprotobuf -lprotobuf-lite -lcbook -lz
LIBS         += -L$(PYTHIA8_DIR)/$(LIBDIRARCH) -lpythia8

SOURCE_FILES1 := $(shell ls -1 main.cc)

INCLUDE1=-I./src
INCLUDE2=-I.
INCLUDE3=-I$(PROMC)/include -I$(PROMC)/src
INCLUDE4=-I$(HEPMC)/include
INCLUDE5=-I$(PYTHIA8_DIR)/include


# build object files
objects1       = $(patsubst %.cc,%.o,$(SOURCE_FILES1))


%.o: %.cc
	$(CXX) $(OPT) $(CXXFLAGS) $(INCLUDE1) $(INCLUDE2) $(INCLUDE3) $(INCLUDE4) $(INCLUDE5) -o $@ -c $<

Tasks:     clean main.exe


LIBOBJS = $(patsubst %.cc,%.o,$(SOURCE_FILES))

main.exe: $(objects1)
	$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@

clean:
	        @rm -f *.o *~ main.exe src/*.o ;  echo "Clear.."
