import os

from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import mm, GeV, MeV, m, deg

SIM = DD4hepSimulation()

## The compact XML file
SIM.compactFile = ""
## Lorentz boost for the crossing angle, in radian!
SIM.crossingAngleBoost = 0.010
SIM.enableDetailedShowerMode = True
SIM.enableG4GPS = False
SIM.enableG4Gun = False
SIM.enableGun = False
## InputFiles for simulation .stdhep, .slcio, .HEPEvt, .hepevt, .hepmc files are supported
SIM.inputFiles = []
## Macro file to execute for runType 'run' or 'vis'
SIM.macroFile = ""
## number of events to simulate, used in batch mode
SIM.numberOfEvents = 0
## Outputfile from the simulation,only lcio output is supported
SIM.outputFile = "dummyOutput.slcio"
## Verbosity use integers from 1(most) to 7(least) verbose
## or strings: VERBOSE, DEBUG, INFO, WARNING, ERROR, FATAL, ALWAYS
SIM.printLevel = 3
## The type of action to do in this invocation
## batch: just simulate some events, needs numberOfEvents, and input file or gun
## vis: enable visualisation, run the macroFile if it is set
## run: run the macroFile and exit
## shell: enable interactive session
SIM.runType = "batch"
## Skip first N events when reading a file
SIM.skipNEvents = 0
## Steering file to change default behaviour
SIM.steeringFile = None
## FourVector of translation for the Smearing of the Vertex position: x y z t
SIM.vertexOffset = [0.0, 0.0, 0.0, 0.0]
## FourVector of the Sigma for the Smearing of the Vertex position: x y z t
SIM.vertexSigma = [0.0, 0.0, 0.0, 0.0]


################################################################################
## Action holding sensitive detector actions
##   The default tracker and calorimeter actions can be set with
##
##   >>> SIM = DD4hepSimulation()
##   >>> SIM.action.tracker = "Geant4TrackerAction"
##   >>> SIM.action.calo    = "Geant4CalorimeterAction"
##
##   for specific subdetectors specific sensitive detectors can be set based on pattern matching
##
##   >>> SIM = DD4hepSimulation()
##   >>> SIM.action.mapActions['tpc'] = "TPCSDAction"
##
##   and additional parameters for the sensitive detectors can be set when the map is given a tuple
##
##   >>> SIM = DD4hepSimulation()
##   >>> SIM.action.mapActions['ecal'] =( "CaloPreShowerSDAction", {"FirstLayerNumber": 1} )
##
##
################################################################################

##  set the default tracker action
SIM.action.tracker = "Geant4TrackerWeightedAction"

##  set the default calorimeter action
SIM.action.calo = "Geant4ScintillatorCalorimeterAction"

##  create a map of patterns and actions to be applied to sensitive detectors
##         example: SIM.action.mapActions['tpc'] = "TPCSDAction"
SIM.action.mapActions = {}


################################################################################
## Configuration for the magnetic field (stepper)
################################################################################
SIM.field.delta_chord = 0.25 * mm
SIM.field.delta_intersection = 0.001 * mm
SIM.field.delta_one_step = 0.01 * mm
SIM.field.eps_max = 0.001 * mm
SIM.field.eps_min = 5e-05 * mm
SIM.field.equation = "Mag_UsualEqRhs"
SIM.field.largest_step = 10.0 * m
SIM.field.min_chord_step = 0.01 * mm
SIM.field.stepper = "ClassicalRK4"


################################################################################
## Configuration for sensitive detector filters
##
##   Set the default filter for tracker or caliromter
##   >>> SIM.filter.tracker = "edep1kev"
##   >>> SIM.filter.calo = ""
##
##   Assign a filter to a sensitive detector via pattern matching
##   >>> SIM.filter.mapDetFilter['FTD'] = "edep1kev"
##
##   Or more than one filter:
##   >>> SIM.filter.mapDetFilter['FTD'] = ["edep1kev", "geantino"]
##
##   Don't use the default filter or anything else:
##   >>> SIM.filter.mapDetFilter['TPC'] = None ## or "" or []
##
##   Create a custom filter. The dictionary is used to instantiate the filter later on
##   >>> SIM.filter.filters['edep3kev'] = dict(name="EnergyDepositMinimumCut/3keV", parameter={"Cut": 3.0*keV} )
##
##
################################################################################

##  default filter for calorimeter sensitive detectors; this is applied if no other filter is used for a calorimeter
SIM.filter.calo = "edep0"

##  list of filter objects: map between name and parameter dictionary
SIM.filter.filters = {
    "edep0": {"parameter": {"Cut": 0.0}, "name": "EnergyDepositMinimumCut/Cut0"},
    "geantino": {"parameter": {}, "name": "GeantinoRejectFilter/GeantinoRejector"},
    "edep1kev": {"parameter": {"Cut": 0.001}, "name": "EnergyDepositMinimumCut"},
}

##  a map between patterns and filter objects, using patterns to attach filters to sensitive detector
SIM.filter.mapDetFilter = {}

##  default filter for tracking sensitive detectors; this is applied if no other filter is used for a tracker
SIM.filter.tracker = "edep1kev"


################################################################################
## Configuration for the DDG4 ParticleGun
################################################################################

##  direction of the particle gun, 3 vector
SIM.gun.direction = (0, 0, 1)

## choose the distribution of the random direction for theta
##
##     Options for random distributions:
##
##     'uniform' is the default distribution, flat in theta
##     'cos(theta)' is flat in cos(theta)
##     'eta', or 'pseudorapidity' is flat in pseudorapity
##     'ffbar' is distributed according to 1+cos^2(theta)
##
##     Setting a distribution will set isotrop = True
##
SIM.gun.distribution = None
SIM.gun.energy = None

##  isotropic distribution for the particle gun
##
##     use the options phiMin, phiMax, thetaMin, and thetaMax to limit the range of randomly distributed directions
##     if one of these options is not None the random distribution will be set to True and cannot be turned off!
##
SIM.gun.isotrop = False
SIM.gun.multiplicity = 1
SIM.gun.particle = "mu-"
SIM.gun.phiMax = None

## Minimal azimuthal angle for random distribution
SIM.gun.phiMin = None

##  position of the particle gun, 3 vector
SIM.gun.position = (0.0, 0.0, 0.0)
SIM.gun.thetaMax = None
SIM.gun.thetaMin = None


################################################################################
## Configuration for the output levels of DDG4 components
################################################################################

## Output level for input sources
SIM.output.inputStage = 3

## Output level for Geant4 kernel
SIM.output.kernel = 3

## Output level for ParticleHandler
SIM.output.part = 3

## Output level for Random Number Generator setup
SIM.output.random = 6


################################################################################
## Configuration for the Particle Handler/ MCTruth treatment
################################################################################

##  Keep all created particles
SIM.part.keepAllParticles = False

## Minimal distance between particle vertex and endpoint of parent after
##     which the vertexIsNotEndpointOfParent flag is set
##
SIM.part.minDistToParentVertex = 2.2e-14

## MinimalKineticEnergy to store particles created in the tracking region
SIM.part.minimalKineticEnergy = 1.0 * MeV

##  Printout at End of Tracking
SIM.part.printEndTracking = False

##  Printout at Start of Tracking
SIM.part.printStartTracking = False

## List of processes to save, on command line give as whitespace separated string in quotation marks
SIM.part.saveProcesses = ["Decay"]


################################################################################
## Configuration for the PhysicsList
################################################################################
SIM.physics.decays = False
SIM.physics.list = "FTFP_BERT"

##  location of particle.tbl file containing extra particles and their lifetime information
##
SIM.physics.pdgfile = os.path.join(os.environ.get("DD4hepINSTALL"), "examples/DDG4/examples/particle.tbl")

##  The global geant4 rangecut for secondary production
##
##     Default is 0.7 mm as is the case in geant4 10
##
##     To disable this plugin and be absolutely sure to use the Geant4 default range cut use "None"
##
##     Set printlevel to DEBUG to see a printout of all range cuts,
##     but this only works if range cut is not "None"
##
SIM.physics.rangecut = 0.7 * mm

SIM.physics.rejectPDGs = {1, 2, 3, 4, 5, 6, 21, 23, 24, 25}

################################################################################
## Properties for the random number generator
################################################################################

## If True, calculate random seed for each event based on eventID and runID
## allows reproducibility even when SkippingEvents
SIM.random.enableEventSeed = True
SIM.random.file = None
SIM.random.luxury = 1
SIM.random.replace_gRandom = True
SIM.random.seed = None
SIM.random.type = None
