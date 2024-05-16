"""
Pythia8, integrated in the FCCSW framework.

Generates according to a pythia .cmd file and saves them in fcc edm format.

"""

import os
from GaudiKernel import SystemOfUnits as units
from Gaudi.Configuration import *

from Configurables import ApplicationMgr

ApplicationMgr().EvtSel = "NONE"
ApplicationMgr().EvtMax = 1000
ApplicationMgr().OutputLevel = INFO
ApplicationMgr().ExtSvc += ["RndmGenSvc"]

#### Data service
from Configurables import k4DataSvc

podioevent = k4DataSvc("EventDataSvc")
ApplicationMgr().ExtSvc += [podioevent]

from Configurables import GaussSmearVertex

smeartool = GaussSmearVertex()
smeartool.xVertexSigma = 0.5 * units.mm
smeartool.yVertexSigma = 0.5 * units.mm
smeartool.zVertexSigma = 40.0 * units.mm
smeartool.tVertexSigma = 180.0 * units.picosecond

from Configurables import PythiaInterface

pythia8gentool = PythiaInterface()
### Example of pythia configuration file to generate events
# take from $K4GEN if defined, locally if not
pythia8gentool.pythiacard = "p8_ee_Z_Ztautau_ecm125.cmd"
pythia8gentool.doEvtGenDecays = False
pythia8gentool.printPythiaStatistics = True

from Configurables import GenAlg

pythia8gen = GenAlg("Pythia8")
pythia8gen.SignalProvider = pythia8gentool
pythia8gen.VertexSmearingTool = smeartool
pythia8gen.hepmc.Path = "hepmc"
ApplicationMgr().TopAlg += [pythia8gen]

### Reads an HepMC::GenEvent from the data service and writes a collection of EDM Particles
from Configurables import HepMCToEDMConverter

hepmc_converter = HepMCToEDMConverter()
ApplicationMgr().TopAlg += [hepmc_converter]

### Filters generated particles
# accept is a list of particle statuses that should be accepted
from Configurables import GenParticleFilter

genfilter = GenParticleFilter("StableParticles")
genfilter.accept = [1]
genfilter.GenParticles.Path = "GenParticles"
genfilter.GenParticlesFiltered.Path = "GenParticlesStable"
ApplicationMgr().TopAlg += [genfilter]

from Configurables import EDMToHepMCConverter

edm_converter = EDMToHepMCConverter("BackConverter")
edm_converter.GenParticles.Path = "GenParticleStable"

from Configurables import HepMCFileWriter

dumper = HepMCFileWriter("Dumper")
dumper.hepmc.Path = "hepmc"
ApplicationMgr().TopAlg += [dumper]

# from Configurables import PodioOutput
# out = PodioOutput("out")
# out.filename = "out_pythia.root"
# out.outputCommands = ["keep *"]
