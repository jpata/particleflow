from Gaudi.Configuration import *
from Gaudi.Configuration import DEBUG, WARNING, INFO

from Configurables import LcioEvent, k4DataSvc, MarlinProcessorWrapper
from k4MarlinWrapper.parseConstants import *

algList = []


CONSTANTS = {
    "BCReco": "3TeV",
}

parseConstants(CONSTANTS)


# For converters
from Configurables import ToolSvc, Lcio2EDM4hepTool, EDM4hep2LcioTool


# read = LcioEvent()
# read.OutputLevel = WARNING
# read.Files = ["ttbar.slcio"]
# algList.append(read)


from Configurables import k4DataSvc, PodioInput

evtsvc = k4DataSvc("EventDataSvc")
evtsvc.input = "$TEST_DIR/inputFiles/ttbar1_edm4hep.root"


inp = PodioInput("InputReader")
inp.collections = [
    "EventHeader",
    "MCParticles",
    "VertexBarrelCollection",
    "VertexEndcapCollection",
    "InnerTrackerBarrelCollection",
    "OuterTrackerBarrelCollection",
    "InnerTrackerEndcapCollection",
    "OuterTrackerEndcapCollection",
    "ECalEndcapCollection",
    "ECalEndcapCollectionContributions",
    "ECalBarrelCollection",
    "ECalBarrelCollectionContributions",
    "ECalPlugCollection",
    "ECalPlugCollectionContributions",
    "HCalBarrelCollection",
    "HCalBarrelCollectionContributions",
    "HCalEndcapCollection",
    "HCalEndcapCollectionContributions",
    "HCalRingCollection",
    "HCalRingCollectionContributions",
    "YokeBarrelCollection",
    "YokeBarrelCollectionContributions",
    "YokeEndcapCollection",
    "YokeEndcapCollectionContributions",
    "LumiCalCollection",
    "LumiCalCollectionContributions",
    "BeamCalCollection",
    "BeamCalCollectionContributions",
]
inp.OutputLevel = DEBUG


MyAIDAProcessor = MarlinProcessorWrapper("MyAIDAProcessor")
MyAIDAProcessor.OutputLevel = WARNING
MyAIDAProcessor.ProcessorType = "AIDAProcessor"
MyAIDAProcessor.Parameters = {"Compress": ["1"], "FileName": ["histograms"], "FileType": ["root"]}

# EDM4hep to LCIO converter
edmConvTool = EDM4hep2LcioTool("EDM4hep2lcio")
edmConvTool.convertAll = True
edmConvTool.OutputLevel = DEBUG
MyAIDAProcessor.EDM4hep2LcioTool = edmConvTool


InitDD4hep = MarlinProcessorWrapper("InitDD4hep")
InitDD4hep.OutputLevel = WARNING
InitDD4hep.ProcessorType = "InitializeDD4hep"
InitDD4hep.Parameters = {
    "DD4hepXMLFile": [os.environ["LCGEO"] + "/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml"],
    "EncodingStringParameter": ["GlobalTrackerReadoutID"],
}

Config = MarlinProcessorWrapper("Config")
Config.OutputLevel = WARNING
Config.ProcessorType = "CLICRecoConfig"
Config.Parameters = {
    "BeamCal": ["3TeV"],
    "BeamCalChoices": ["3TeV", "380GeV"],
    "Overlay": ["False"],
    "OverlayChoices": [
        "False",
        "350GeV_CDR",
        "350GeV",
        "350GeV_L6",
        "380GeV",
        "380GeV_CDR",
        "380GeV_L6",
        "420GeV",
        "500GeV",
        "1.4TeV",
        "3TeV",
        "3TeV_L6",
    ],
    "Tracking": ["Conformal"],
    "TrackingChoices": ["Truth", "Conformal"],
    "VertexUnconstrained": ["OFF"],
    "VertexUnconstrainedChoices": ["ON", "OFF"],
}

VXDBarrelDigitiser = MarlinProcessorWrapper("VXDBarrelDigitiser")
VXDBarrelDigitiser.OutputLevel = WARNING
VXDBarrelDigitiser.ProcessorType = "DDPlanarDigiProcessor"
VXDBarrelDigitiser.Parameters = {
    "IsStrip": ["false"],
    "ResolutionU": ["0.003", "0.003", "0.003", "0.003", "0.003", "0.003"],
    "ResolutionV": ["0.003", "0.003", "0.003", "0.003", "0.003", "0.003"],
    "SimTrackHitCollectionName": ["VertexBarrelCollection"],
    "SimTrkHitRelCollection": ["VXDTrackerHitRelations"],
    "SubDetectorName": ["Vertex"],
    "TrackerHitCollectionName": ["VXDTrackerHits"],
}

# LCIO to EDM4hep
VXDBarrelDigitiserLCIOConv = Lcio2EDM4hepTool("VXDBarrelDigitiserLCIOConv")
VXDBarrelDigitiserLCIOConv.convertAll = False
VXDBarrelDigitiserLCIOConv.collNameMapping = {
    # This should be a TrackerHitPlane, but it gets treated as a TrackerHit
    "VXDTrackerHits": "VXDTrackerHits",
    "VXDTrackerHitRelations": "VXDTrackerHitRelations",
}
VXDBarrelDigitiserLCIOConv.OutputLevel = DEBUG
# Add it to VXDBarrelDigitiser Algorithm
VXDBarrelDigitiser.Lcio2EDM4hepTool = VXDBarrelDigitiserLCIOConv


VXDEndcapDigitiser = MarlinProcessorWrapper("VXDEndcapDigitiser")
VXDEndcapDigitiser.OutputLevel = WARNING
VXDEndcapDigitiser.ProcessorType = "DDPlanarDigiProcessor"
VXDEndcapDigitiser.Parameters = {
    "IsStrip": ["false"],
    "ResolutionU": ["0.003", "0.003", "0.003", "0.003", "0.003", "0.003"],
    "ResolutionV": ["0.003", "0.003", "0.003", "0.003", "0.003", "0.003"],
    "SimTrackHitCollectionName": ["VertexEndcapCollection"],
    "SimTrkHitRelCollection": ["VXDEndcapTrackerHitRelations"],
    "SubDetectorName": ["Vertex"],
    "TrackerHitCollectionName": ["VXDEndcapTrackerHits"],
}

# LCIO to EDM4hep
VXDEndcapDigitiserLCIOConv = Lcio2EDM4hepTool("VXDEndcapDigitiserLCIOConv")
VXDEndcapDigitiserLCIOConv.convertAll = False
VXDEndcapDigitiserLCIOConv.collNameMapping = {
    # This should be a TrackerHitPlane, but it gets treated as a TrackerHit
    "VXDEndcapTrackerHits": "VXDEndcapTrackerHits",
    "VXDEndcapTrackerHitRelations": "VXDEndcapTrackerHitRelations",
}
VXDEndcapDigitiserLCIOConv.OutputLevel = DEBUG
# Add it to VXDEndcapDigitiser Algorithm
VXDEndcapDigitiser.Lcio2EDM4hepTool = VXDEndcapDigitiserLCIOConv


InnerPlanarDigiProcessor = MarlinProcessorWrapper("InnerPlanarDigiProcessor")
InnerPlanarDigiProcessor.OutputLevel = WARNING
InnerPlanarDigiProcessor.ProcessorType = "DDPlanarDigiProcessor"
InnerPlanarDigiProcessor.Parameters = {
    "IsStrip": ["false"],
    "ResolutionU": ["0.007"],
    "ResolutionV": ["0.09"],
    "SimTrackHitCollectionName": ["InnerTrackerBarrelCollection"],
    "SimTrkHitRelCollection": ["InnerTrackerBarrelHitsRelations"],
    "SubDetectorName": ["InnerTrackers"],
    "TrackerHitCollectionName": ["ITrackerHits"],
}

# LCIO to EDM4hep
InnerPlanarDigiProcessorLCIOConv = Lcio2EDM4hepTool("InnerPlanarDigiProcessorLCIOConv")
InnerPlanarDigiProcessorLCIOConv.convertAll = False
InnerPlanarDigiProcessorLCIOConv.collNameMapping = {
    # This should be a TrackerHitPlane, but it gets treated as a TrackerHit
    "ITrackerHits": "ITrackerHits",
    "InnerTrackerBarrelHitsRelations": "InnerTrackerBarrelHitsRelations",
}
InnerPlanarDigiProcessorLCIOConv.OutputLevel = DEBUG
# Add it to InnerPlanarDigiProcessor Algorithm
InnerPlanarDigiProcessor.Lcio2EDM4hepTool = InnerPlanarDigiProcessorLCIOConv


InnerEndcapPlanarDigiProcessor = MarlinProcessorWrapper("InnerEndcapPlanarDigiProcessor")
InnerEndcapPlanarDigiProcessor.OutputLevel = WARNING
InnerEndcapPlanarDigiProcessor.ProcessorType = "DDPlanarDigiProcessor"
InnerEndcapPlanarDigiProcessor.Parameters = {
    "IsStrip": ["false"],
    "ResolutionU": ["0.005", "0.007", "0.007", "0.007", "0.007", "0.007", "0.007"],
    "ResolutionV": ["0.005", "0.09", "0.09", "0.09", "0.09", "0.09", "0.09"],
    "SimTrackHitCollectionName": ["InnerTrackerEndcapCollection"],
    "SimTrkHitRelCollection": ["InnerTrackerEndcapHitsRelations"],
    "SubDetectorName": ["InnerTrackers"],
    "TrackerHitCollectionName": ["ITrackerEndcapHits"],
}
# LCIO to EDM4hep
InnerEndcapPlanarDigiProcessorLCIOConv = Lcio2EDM4hepTool("InnerEndcapPlanarDigiProcessorLCIOConv")
InnerEndcapPlanarDigiProcessorLCIOConv.convertAll = False
InnerEndcapPlanarDigiProcessorLCIOConv.collNameMapping = {
    # This should be a TrackerHitPlane, but it gets treated as a TrackerHit
    "ITrackerEndcapHits": "ITrackerEndcapHits",
    "InnerTrackerEndcapHitsRelations": "InnerTrackerEndcapHitsRelations",
}
InnerEndcapPlanarDigiProcessorLCIOConv.OutputLevel = DEBUG
# Add it to InnerEndcapPlanarDigiProcessor Algorithm
InnerEndcapPlanarDigiProcessor.Lcio2EDM4hepTool = InnerEndcapPlanarDigiProcessorLCIOConv


OuterPlanarDigiProcessor = MarlinProcessorWrapper("OuterPlanarDigiProcessor")
OuterPlanarDigiProcessor.OutputLevel = WARNING
OuterPlanarDigiProcessor.ProcessorType = "DDPlanarDigiProcessor"
OuterPlanarDigiProcessor.Parameters = {
    "IsStrip": ["false"],
    "ResolutionU": ["0.007", "0.007", "0.007"],
    "ResolutionV": ["0.09", "0.09", "0.09"],
    "SimTrackHitCollectionName": ["OuterTrackerBarrelCollection"],
    "SimTrkHitRelCollection": ["OuterTrackerBarrelHitsRelations"],
    "SubDetectorName": ["OuterTrackers"],
    "TrackerHitCollectionName": ["OTrackerHits"],
}
# LCIO to EDM4hep
OuterPlanarDigiProcessorLCIOConv = Lcio2EDM4hepTool("OuterPlanarDigiProcessorLCIOConv")
OuterPlanarDigiProcessorLCIOConv.convertAll = False
OuterPlanarDigiProcessorLCIOConv.collNameMapping = {
    # This should be a TrackerHitPlane, but it gets treated as a TrackerHit
    "OTrackerHits": "OTrackerHits",
    "OuterTrackerBarrelHitsRelations": "OuterTrackerBarrelHitsRelations",
}
OuterPlanarDigiProcessorLCIOConv.OutputLevel = DEBUG
# Add it to OuterPlanarDigiProcessor Algorithm
OuterPlanarDigiProcessor.Lcio2EDM4hepTool = OuterPlanarDigiProcessorLCIOConv


OuterEndcapPlanarDigiProcessor = MarlinProcessorWrapper("OuterEndcapPlanarDigiProcessor")
OuterEndcapPlanarDigiProcessor.OutputLevel = WARNING
OuterEndcapPlanarDigiProcessor.ProcessorType = "DDPlanarDigiProcessor"
OuterEndcapPlanarDigiProcessor.Parameters = {
    "IsStrip": ["false"],
    "ResolutionU": ["0.007", "0.007", "0.007", "0.007", "0.007"],
    "ResolutionV": ["0.09", "0.09", "0.09", "0.09", "0.09"],
    "SimTrackHitCollectionName": ["OuterTrackerEndcapCollection"],
    "SimTrkHitRelCollection": ["OuterTrackerEndcapHitsRelations"],
    "SubDetectorName": ["OuterTrackers"],
    "TrackerHitCollectionName": ["OTrackerEndcapHits"],
}
# LCIO to EDM4hep
OuterEndcapPlanarDigiProcessorLCIOConv = Lcio2EDM4hepTool("OuterEndcapPlanarDigiProcessorLCIOConv")
OuterEndcapPlanarDigiProcessorLCIOConv.convertAll = False
OuterEndcapPlanarDigiProcessorLCIOConv.collNameMapping = {
    # This should be a TrackerHitPlane, but it gets treated as a TrackerHit
    "OTrackerEndcapHits": "OTrackerEndcapHits",
    "OuterTrackerEndcapHitsRelations": "OuterTrackerEndcapHitsRelations",
}
OuterEndcapPlanarDigiProcessorLCIOConv.OutputLevel = DEBUG
# Add it to OuterEndcapPlanarDigiProcessor Algorithm
OuterEndcapPlanarDigiProcessor.Lcio2EDM4hepTool = OuterEndcapPlanarDigiProcessorLCIOConv


MyTruthTrackFinder = MarlinProcessorWrapper("MyTruthTrackFinder")
MyTruthTrackFinder.OutputLevel = WARNING
MyTruthTrackFinder.ProcessorType = "TruthTrackFinder"
MyTruthTrackFinder.Parameters = {
    "FitForward": ["true"],
    "MCParticleCollectionName": ["MCParticle"],
    "SiTrackCollectionName": ["SiTracks"],
    "SiTrackRelationCollectionName": ["SiTrackRelations"],
    "SimTrackerHitRelCollectionNames": [
        "VXDTrackerHitRelations",
        "InnerTrackerBarrelHitsRelations",
        "OuterTrackerBarrelHitsRelations",
        "VXDEndcapTrackerHitRelations",
        "InnerTrackerEndcapHitsRelations",
        "OuterTrackerEndcapHitsRelations",
    ],
    "TrackerHitCollectionNames": [
        "VXDTrackerHits",
        "ITrackerHits",
        "OTrackerHits",
        "VXDEndcapTrackerHits",
        "ITrackerEndcapHits",
        "OTrackerEndcapHits",
    ],
    "UseTruthInPrefit": ["false"],
}

MyConformalTracking = MarlinProcessorWrapper("MyConformalTracking")
MyConformalTracking.OutputLevel = WARNING
MyConformalTracking.ProcessorType = "ConformalTrackingV2"
MyConformalTracking.Parameters = {
    "DebugHits": ["DebugHits"],
    "DebugPlots": ["false"],
    "DebugTiming": ["false"],
    "MCParticleCollectionName": ["MCParticle"],
    "MaxHitInvertedFit": ["0"],
    "MinClustersOnTrackAfterFit": ["3"],
    "RelationsNames": [
        "VXDTrackerHitRelations",
        "VXDEndcapTrackerHitRelations",
        "InnerTrackerBarrelHitsRelations",
        "OuterTrackerBarrelHitsRelations",
        "InnerTrackerEndcapHitsRelations",
        "OuterTrackerEndcapHitsRelations",
    ],
    "RetryTooManyTracks": ["true"],
    "SiTrackCollectionName": ["SiTracksCT"],
    "SortTreeResults": ["true"],
    "Steps": [
        "[VXDBarrel]",
        "@Collections",
        ":",
        "VXDTrackerHits",
        "@Parameters",
        ":",
        "MaxCellAngle",
        ":",
        "0.005;",
        "MaxCellAngleRZ",
        ":",
        "0.005;",
        "Chi2Cut",
        ":",
        "100;",
        "MinClustersOnTrack",
        ":",
        "4;",
        "MaxDistance",
        ":",
        "0.02;",
        "SlopeZRange:",
        "10.0;",
        "HighPTCut:",
        "10.0;",
        "@Flags",
        ":",
        "HighPTFit,",
        "VertexToTracker",
        "@Functions",
        ":",
        "CombineCollections,",
        "BuildNewTracks",
        "[VXDEncap]",
        "@Collections",
        ":",
        "VXDEndcapTrackerHits",
        "@Parameters",
        ":",
        "MaxCellAngle",
        ":",
        "0.005;",
        "MaxCellAngleRZ",
        ":",
        "0.005;",
        "Chi2Cut",
        ":",
        "100;",
        "MinClustersOnTrack",
        ":",
        "4;",
        "MaxDistance",
        ":",
        "0.02;",
        "SlopeZRange:",
        "10.0;",
        "HighPTCut:",
        "0.0;",
        "@Flags",
        ":",
        "HighPTFit,",
        "VertexToTracker",
        "@Functions",
        ":",
        "CombineCollections,",
        "ExtendTracks",
        "[LowerCellAngle1]",
        "@Collections",
        ":",
        "VXDTrackerHits,",
        "VXDEndcapTrackerHits",
        "@Parameters",
        ":",
        "MaxCellAngle",
        ":",
        "0.025;",
        "MaxCellAngleRZ",
        ":",
        "0.025;",
        "Chi2Cut",
        ":",
        "100;",
        "MinClustersOnTrack",
        ":",
        "4;",
        "MaxDistance",
        ":",
        "0.02;",
        "SlopeZRange:",
        "10.0;",
        "HighPTCut:",
        "10.0;",
        "@Flags",
        ":",
        "HighPTFit,",
        "VertexToTracker,",
        "RadialSearch",
        "@Functions",
        ":",
        "CombineCollections,",
        "BuildNewTracks",
        "[LowerCellAngle2]",
        "@Collections",
        ":",
        "@Parameters",
        ":",
        "MaxCellAngle",
        ":",
        "0.05;",
        "MaxCellAngleRZ",
        ":",
        "0.05;",
        "Chi2Cut",
        ":",
        "2000;",
        "MinClustersOnTrack",
        ":",
        "4;",
        "MaxDistance",
        ":",
        "0.02;",
        "SlopeZRange:",
        "10.0;",
        "HighPTCut:",
        "10.0;",
        "@Flags",
        ":",
        "HighPTFit,",
        "VertexToTracker,",
        "RadialSearch",
        "@Functions",
        ":",
        "BuildNewTracks,",
        "SortTracks",
        "[Tracker]",
        "@Collections",
        ":",
        "ITrackerHits,",
        "OTrackerHits,",
        "ITrackerEndcapHits,",
        "OTrackerEndcapHits",
        "@Parameters",
        ":",
        "MaxCellAngle",
        ":",
        "0.05;",
        "MaxCellAngleRZ",
        ":",
        "0.05;",
        "Chi2Cut",
        ":",
        "2000;",
        "MinClustersOnTrack",
        ":",
        "4;",
        "MaxDistance",
        ":",
        "0.02;",
        "SlopeZRange:",
        "10.0;",
        "HighPTCut:",
        "0.0;",
        "@Flags",
        ":",
        "HighPTFit,",
        "VertexToTracker,",
        "RadialSearch",
        "@Functions",
        ":",
        "CombineCollections,",
        "ExtendTracks",
        "[Displaced]",
        "@Collections",
        ":",
        "VXDTrackerHits,",
        "VXDEndcapTrackerHits,",
        "ITrackerHits,",
        "OTrackerHits,",
        "ITrackerEndcapHits,",
        "OTrackerEndcapHits",
        "@Parameters",
        ":",
        "MaxCellAngle",
        ":",
        "0.05;",
        "MaxCellAngleRZ",
        ":",
        "0.05;",
        "Chi2Cut",
        ":",
        "1000;",
        "MinClustersOnTrack",
        ":",
        "5;",
        "MaxDistance",
        ":",
        "0.015;",
        "SlopeZRange:",
        "10.0;",
        "HighPTCut:",
        "10.0;",
        "@Flags",
        ":",
        "OnlyZSchi2cut,",
        "RadialSearch",
        "@Functions",
        ":",
        "CombineCollections,",
        "BuildNewTracks",
    ],
    "ThetaRange": ["0.05"],
    "TooManyTracks": ["400000"],
    "TrackerHitCollectionNames": [
        "VXDTrackerHits",
        "VXDEndcapTrackerHits",
        "ITrackerHits",
        "OTrackerHits",
        "ITrackerEndcapHits",
        "OTrackerEndcapHits",
    ],
    "trackPurity": ["0.7"],
}
# LCIO to EDM4hep
MyConformalTrackingLCIOConv = Lcio2EDM4hepTool("MyConformalTrackingLCIOConv")
MyConformalTrackingLCIOConv.convertAll = False
MyConformalTrackingLCIOConv.collNameMapping = {
    # This should be a TrackerHitPlane, but it gets treated as a TrackerHit
    "DebugHits": "DebugHits",
    "SiTracksCT": "SiTracksCT",
}
MyConformalTrackingLCIOConv.OutputLevel = DEBUG
# Add it to MyConformalTracking Algorithm
MyConformalTracking.Lcio2EDM4hepTool = MyConformalTrackingLCIOConv


ClonesAndSplitTracksFinder = MarlinProcessorWrapper("ClonesAndSplitTracksFinder")
ClonesAndSplitTracksFinder.OutputLevel = WARNING
ClonesAndSplitTracksFinder.ProcessorType = "ClonesAndSplitTracksFinder"
ClonesAndSplitTracksFinder.Parameters = {
    "EnergyLossOn": ["true"],
    "InputTrackCollectionName": ["SiTracksCT"],
    "MultipleScatteringOn": ["true"],
    "OutputTrackCollectionName": ["SiTracks"],
    "SmoothOn": ["false"],
    "extrapolateForward": ["true"],
    "maxSignificancePhi": ["3"],
    "maxSignificancePt": ["2"],
    "maxSignificanceTheta": ["3"],
    "mergeSplitTracks": ["true"],
    "minTrackPt": ["1"],
}
# LCIO to EDM4hep
ClonesAndSplitTracksFinderLCIOConv = Lcio2EDM4hepTool("ClonesAndSplitTracksFinderLCIOConv")
ClonesAndSplitTracksFinderLCIOConv.convertAll = False
ClonesAndSplitTracksFinderLCIOConv.collNameMapping = {"SiTracks": "SiTracks"}
ClonesAndSplitTracksFinderLCIOConv.OutputLevel = DEBUG
# Add it to ClonesAndSplitTracksFinder Algorithm
ClonesAndSplitTracksFinder.Lcio2EDM4hepTool = ClonesAndSplitTracksFinderLCIOConv


Refit = MarlinProcessorWrapper("Refit")
Refit.OutputLevel = WARNING
Refit.ProcessorType = "RefitFinal"
Refit.Parameters = {
    "EnergyLossOn": ["true"],
    "InputRelationCollectionName": ["SiTrackRelations"],
    "InputTrackCollectionName": ["SiTracks"],
    "Max_Chi2_Incr": ["1.79769e+30"],
    "MinClustersOnTrackAfterFit": ["3"],
    "MultipleScatteringOn": ["true"],
    "OutputRelationCollectionName": ["SiTracks_Refitted_Relation"],
    "OutputTrackCollectionName": ["SiTracks_Refitted"],
    "ReferencePoint": ["-1"],
    "SmoothOn": ["false"],
    "extrapolateForward": ["true"],
}
# LCIO to EDM4hep
RefitLCIOConv = Lcio2EDM4hepTool("Refit")
RefitLCIOConv.convertAll = False
RefitLCIOConv.collNameMapping = {"SiTracks_Refitted": "SiTracks_Refitted"}
RefitLCIOConv.OutputLevel = DEBUG
# Add it to RefitLCIOConv Algorithm
Refit.Lcio2EDM4hepTool = RefitLCIOConv


MyClicEfficiencyCalculator = MarlinProcessorWrapper("MyClicEfficiencyCalculator")
MyClicEfficiencyCalculator.OutputLevel = WARNING
MyClicEfficiencyCalculator.ProcessorType = "ClicEfficiencyCalculator"
MyClicEfficiencyCalculator.Parameters = {
    "MCParticleCollectionName": ["MCParticle"],
    "MCParticleNotReco": ["MCParticleNotReco"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "TrackCollectionName": ["SiTracks_Refitted"],
    "TrackerHitCollectionNames": [
        "VXDTrackerHits",
        "VXDEndcapTrackerHits",
        "ITrackerHits",
        "OTrackerHits",
        "ITrackerEndcapHits",
        "OTrackerEndcapHits",
    ],
    "TrackerHitRelCollectionNames": [
        "VXDTrackerHitRelations",
        "VXDEndcapTrackerHitRelations",
        "InnerTrackerBarrelHitsRelations",
        "OuterTrackerBarrelHitsRelations",
        "InnerTrackerEndcapHitsRelations",
        "OuterTrackerEndcapHitsRelations",
    ],
    "efficiencyTreeName": ["trktree"],
    "mcTreeName": ["mctree"],
    "morePlots": ["false"],
    "purityTreeName": ["puritytree"],
    "reconstructableDefinition": ["ILDLike"],
    "vertexBarrelID": ["1"],
}
# LCIO to EDM4hep
MyClicEfficiencyCalculatorLCIOConv = Lcio2EDM4hepTool("MyClicEfficiencyCalculator")
MyClicEfficiencyCalculatorLCIOConv.convertAll = False
MyClicEfficiencyCalculatorLCIOConv.collNameMapping = {"MCParticleNotReco": "MCParticleNotReco"}
MyClicEfficiencyCalculatorLCIOConv.OutputLevel = DEBUG
# Add it to MyClicEfficiencyCalculatorLCIOConv Algorithm
MyClicEfficiencyCalculator.Lcio2EDM4hepTool = MyClicEfficiencyCalculatorLCIOConv


MyTrackChecker = MarlinProcessorWrapper("MyTrackChecker")
MyTrackChecker.OutputLevel = WARNING
MyTrackChecker.ProcessorType = "TrackChecker"
MyTrackChecker.Parameters = {
    "MCParticleCollectionName": ["MCParticle"],
    "TrackCollectionName": ["SiTracks_Refitted"],
    "TrackRelationCollectionName": ["SiTracksMCTruthLink"],
    "TreeName": ["checktree"],
    "UseOnlyTree": ["true"],
}


EventNumber = MarlinProcessorWrapper("EventNumber")
EventNumber.OutputLevel = WARNING
EventNumber.ProcessorType = "Statusmonitor"
EventNumber.Parameters = {"HowOften": ["1"]}

MyDDCaloDigi = MarlinProcessorWrapper("MyDDCaloDigi")
MyDDCaloDigi.OutputLevel = WARNING
MyDDCaloDigi.ProcessorType = "DDCaloDigi"
MyDDCaloDigi.Parameters = {
    "CalibECALMIP": ["0.0001"],
    "CalibHCALMIP": ["0.0001"],
    "CalibrECAL": ["35.8411424188", "35.8411424188"],
    "CalibrHCALBarrel": ["49.2031079063"],
    "CalibrHCALEndcap": ["53.6263377733"],
    "CalibrHCALOther": ["62.2125698179"],
    "ECALBarrelTimeWindowMax": ["10"],
    "ECALCollections": ["ECalBarrelCollection", "ECalEndcapCollection", "ECalPlugCollection"],
    "ECALCorrectTimesForPropagation": ["1"],
    "ECALDeltaTimeHitResolution": ["10"],
    "ECALEndcapCorrectionFactor": ["1.0672142727"],
    "ECALEndcapTimeWindowMax": ["10"],
    "ECALGapCorrection": ["1"],
    "ECALGapCorrectionFactor": ["1"],
    "ECALLayers": ["41", "100"],
    "ECALModuleGapCorrectionFactor": ["0.0"],
    "ECALOutputCollection0": ["ECALBarrel"],
    "ECALOutputCollection1": ["ECALEndcap"],
    "ECALOutputCollection2": ["ECALOther"],
    "ECALSimpleTimingCut": ["true"],
    "ECALThreshold": ["5e-05"],
    "ECALThresholdUnit": ["GeV"],
    "ECALTimeResolution": ["10"],
    "ECALTimeWindowMin": ["-1"],
    "ECAL_PPD_N_Pixels": ["10000"],
    "ECAL_PPD_N_Pixels_uncertainty": ["0.05"],
    "ECAL_PPD_PE_per_MIP": ["7"],
    "ECAL_apply_realistic_digi": ["0"],
    "ECAL_deadCellRate": ["0"],
    "ECAL_deadCell_memorise": ["false"],
    "ECAL_default_layerConfig": ["000000000000000"],
    "ECAL_elec_noise_mips": ["0"],
    "ECAL_maxDynamicRange_MIP": ["2500"],
    "ECAL_miscalibration_correl": ["0"],
    "ECAL_miscalibration_uncorrel": ["0"],
    "ECAL_miscalibration_uncorrel_memorise": ["false"],
    "ECAL_pixel_spread": ["0.05"],
    "ECAL_strip_absorbtionLength": ["1e+06"],
    "HCALBarrelTimeWindowMax": ["10"],
    "HCALCollections": ["HCalBarrelCollection", "HCalEndcapCollection", "HCalRingCollection"],
    "HCALCorrectTimesForPropagation": ["1"],
    "HCALDeltaTimeHitResolution": ["10"],
    "HCALEndcapCorrectionFactor": ["1.000"],
    "HCALEndcapTimeWindowMax": ["10"],
    "HCALGapCorrection": ["1"],
    "HCALLayers": ["100"],
    "HCALModuleGapCorrectionFactor": ["0.5"],
    "HCALOutputCollection0": ["HCALBarrel"],
    "HCALOutputCollection1": ["HCALEndcap"],
    "HCALOutputCollection2": ["HCALOther"],
    "HCALSimpleTimingCut": ["true"],
    "HCALThreshold": ["0.00025"],
    "HCALThresholdUnit": ["GeV"],
    "HCALTimeResolution": ["10"],
    "HCALTimeWindowMin": ["-1"],
    "HCAL_PPD_N_Pixels": ["400"],
    "HCAL_PPD_N_Pixels_uncertainty": ["0.05"],
    "HCAL_PPD_PE_per_MIP": ["10"],
    "HCAL_apply_realistic_digi": ["0"],
    "HCAL_deadCellRate": ["0"],
    "HCAL_deadCell_memorise": ["false"],
    "HCAL_elec_noise_mips": ["0"],
    "HCAL_maxDynamicRange_MIP": ["200"],
    "HCAL_miscalibration_correl": ["0"],
    "HCAL_miscalibration_uncorrel": ["0"],
    "HCAL_miscalibration_uncorrel_memorise": ["false"],
    "HCAL_pixel_spread": ["0"],
    "Histograms": ["0"],
    "IfDigitalEcal": ["0"],
    "IfDigitalHcal": ["0"],
    "MapsEcalCorrection": ["0"],
    "RelationOutputCollection": ["RelationCaloHit"],
    "RootFile": ["Digi_SiW.root"],
    "StripEcal_default_nVirtualCells": ["9"],
    "UseEcalTiming": ["1"],
    "UseHcalTiming": ["1"],
    "energyPerEHpair": ["3.6"],
}
# LCIO to EDM4hep
MyDDCaloDigiLCIOConv = Lcio2EDM4hepTool("MyDDCaloDigiLCIOConv")
MyDDCaloDigiLCIOConv.convertAll = False
MyDDCaloDigiLCIOConv.collNameMapping = {
    "ECALBarrel": "ECALBarrel",
    "ECALEndcap": "ECALEndcap",
    "ECALOther": "ECALOther",
    "HCALBarrel": "HCALBarrel",
    "HCALEndcap": "HCALEndcap",
    "HCALOther": "HCALOther",
    "RelationCaloHit": "RelationCaloHit",
}
MyDDCaloDigiLCIOConv.OutputLevel = DEBUG
# Add it to MyDDCaloDigi Algorithm
MyDDCaloDigi.Lcio2EDM4hepTool = MyDDCaloDigiLCIOConv


MyDDMarlinPandora = MarlinProcessorWrapper("MyDDMarlinPandora")
MyDDMarlinPandora.OutputLevel = WARNING
MyDDMarlinPandora.ProcessorType = "DDPandoraPFANewProcessor"
MyDDMarlinPandora.Parameters = {
    "ClusterCollectionName": ["PandoraClusters"],
    "CreateGaps": ["false"],
    "CurvatureToMomentumFactor": ["0.00015"],
    "D0TrackCut": ["200"],
    "D0UnmatchedVertexTrackCut": ["5"],
    "DigitalMuonHits": ["0"],
    "ECalBarrelNormalVector": ["0", "0", "1"],
    "ECalCaloHitCollections": ["ECALBarrel", "ECALEndcap", "ECALOther"],
    "ECalMipThreshold": ["0.5"],
    "ECalScMipThreshold": ["0"],
    "ECalScToEMGeVCalibration": ["1"],
    "ECalScToHadGeVCalibrationBarrel": ["1"],
    "ECalScToHadGeVCalibrationEndCap": ["1"],
    "ECalScToMipCalibration": ["1"],
    "ECalSiMipThreshold": ["0"],
    "ECalSiToEMGeVCalibration": ["1"],
    "ECalSiToHadGeVCalibrationBarrel": ["1"],
    "ECalSiToHadGeVCalibrationEndCap": ["1"],
    "ECalSiToMipCalibration": ["1"],
    "ECalToEMGeVCalibration": ["1.02373335516"],
    "ECalToHadGeVCalibrationBarrel": ["1.24223718397"],
    "ECalToHadGeVCalibrationEndCap": ["1.24223718397"],
    "ECalToMipCalibration": ["181.818"],
    "EMConstantTerm": ["0.01"],
    "EMStochasticTerm": ["0.17"],
    "FinalEnergyDensityBin": ["110."],
    "HCalBarrelNormalVector": ["0", "0", "1"],
    "HCalCaloHitCollections": ["HCALBarrel", "HCALEndcap", "HCALOther"],
    "HCalMipThreshold": ["0.3"],
    "HCalToEMGeVCalibration": ["1.02373335516"],
    "HCalToHadGeVCalibration": ["1.01799349172"],
    "HCalToMipCalibration": ["40.8163"],
    "HadConstantTerm": ["0.03"],
    "HadStochasticTerm": ["0.6"],
    "InputEnergyCorrectionPoints": [],
    "KinkVertexCollections": ["KinkVertices"],
    "LCalCaloHitCollections": [],
    "LHCalCaloHitCollections": [],
    "LayersFromEdgeMaxRearDistance": ["250"],
    "MCParticleCollections": ["MCParticle"],
    "MaxBarrelTrackerInnerRDistance": ["200"],
    "MaxClusterEnergyToApplySoftComp": ["2000."],
    "MaxHCalHitHadronicEnergy": ["1000000"],
    "MaxTrackHits": ["5000"],
    "MaxTrackSigmaPOverP": ["0.15"],
    "MinBarrelTrackerHitFractionOfExpected": ["0"],
    "MinCleanCorrectedHitEnergy": ["0.1"],
    "MinCleanHitEnergy": ["0.5"],
    "MinCleanHitEnergyFraction": ["0.01"],
    "MinFtdHitsForBarrelTrackerHitFraction": ["0"],
    "MinFtdTrackHits": ["0"],
    "MinMomentumForTrackHitChecks": ["0"],
    "MinTpcHitFractionOfExpected": ["0"],
    "MinTrackECalDistanceFromIp": ["0"],
    "MinTrackHits": ["0"],
    "MuonBarrelBField": ["-1.5"],
    "MuonCaloHitCollections": ["MUON"],
    "MuonEndCapBField": ["0.01"],
    "MuonHitEnergy": ["0.5"],
    "MuonToMipCalibration": ["19607.8"],
    "NEventsToSkip": ["0"],
    "NOuterSamplingLayers": ["3"],
    "OutputEnergyCorrectionPoints": [],
    "PFOCollectionName": ["PandoraPFOs"],
    "PandoraSettingsXmlFile": ["PandoraSettings/PandoraSettingsDefault.xml"],
    "ProngVertexCollections": ["ProngVertices"],
    "ReachesECalBarrelTrackerOuterDistance": ["-100"],
    "ReachesECalBarrelTrackerZMaxDistance": ["-50"],
    "ReachesECalFtdZMaxDistance": ["1"],
    "ReachesECalMinFtdLayer": ["0"],
    "ReachesECalNBarrelTrackerHits": ["0"],
    "ReachesECalNFtdHits": ["0"],
    "RelCaloHitCollections": ["RelationCaloHit", "RelationMuonHit"],
    "RelTrackCollections": ["SiTracks_Refitted_Relation"],
    "ShouldFormTrackRelationships": ["1"],
    "SoftwareCompensationEnergyDensityBins": [
        "0",
        "2.",
        "5.",
        "7.5",
        "9.5",
        "13.",
        "16.",
        "20.",
        "23.5",
        "28.",
        "33.",
        "40.",
        "50.",
        "75.",
        "100.",
    ],
    "SoftwareCompensationWeights": [
        "1.61741",
        "-0.00444385",
        "2.29683e-05",
        "-0.0731236",
        "-0.00157099",
        "-7.09546e-07",
        "0.868443",
        "1.0561",
        "-0.0238574",
    ],
    "SplitVertexCollections": ["SplitVertices"],
    "StartVertexAlgorithmName": ["PandoraPFANew"],
    "StartVertexCollectionName": ["PandoraStartVertices"],
    "StripSplittingOn": ["0"],
    "TrackCollections": ["SiTracks_Refitted"],
    "TrackCreatorName": ["DDTrackCreatorCLIC"],
    "TrackStateTolerance": ["0"],
    "TrackSystemName": ["DDKalTest"],
    "UnmatchedVertexTrackMaxEnergy": ["5"],
    "UseEcalScLayers": ["0"],
    "UseNonVertexTracks": ["1"],
    "UseOldTrackStateCalculation": ["0"],
    "UseUnmatchedNonVertexTracks": ["0"],
    "UseUnmatchedVertexTracks": ["1"],
    "V0VertexCollections": ["V0Vertices"],
    "YokeBarrelNormalVector": ["0", "0", "1"],
    "Z0TrackCut": ["200"],
    "Z0UnmatchedVertexTrackCut": ["5"],
    "ZCutForNonVertexTracks": ["250"],
}
# LCIO to EDM4hep
MyDDMarlinPandoraLCIOConv = Lcio2EDM4hepTool("MyDDMarlinPandoraLCIOConv")
MyDDMarlinPandoraLCIOConv.convertAll = False
MyDDMarlinPandoraLCIOConv.collNameMapping = {
    "PandoraClusters": "PandoraClusters",
    "PandoraPFOs": "PandoraPFOs",
    "PandoraStartVertices": "PandoraStartVertices",
}
MyDDMarlinPandoraLCIOConv.OutputLevel = DEBUG
# Add it to MyDDMarlinPandora Algorithm
MyDDMarlinPandora.Lcio2EDM4hepTool = MyDDMarlinPandoraLCIOConv


MyDDSimpleMuonDigi = MarlinProcessorWrapper("MyDDSimpleMuonDigi")
MyDDSimpleMuonDigi.OutputLevel = WARNING
MyDDSimpleMuonDigi.ProcessorType = "DDSimpleMuonDigi"
MyDDSimpleMuonDigi.Parameters = {
    "CalibrMUON": ["70.1"],
    "MUONCollections": ["YokeBarrelCollection", "YokeEndcapCollection"],
    "MUONOutputCollection": ["MUON"],
    "MaxHitEnergyMUON": ["2.0"],
    "MuonThreshold": ["1e-06"],
    "RelationOutputCollection": ["RelationMuonHit"],
}
# LCIO to EDM4hep
MyDDSimpleMuonDigiLCIOConv = Lcio2EDM4hepTool("MyDDSimpleMuonDigiLCIOConv")
MyDDSimpleMuonDigiLCIOConv.convertAll = False
MyDDSimpleMuonDigiLCIOConv.collNameMapping = {"MUON": "MUON", "RelationMuonHit": "RelationMuonHit"}
MyDDSimpleMuonDigiLCIOConv.OutputLevel = DEBUG
# Add it to MyDDSimpleMuonDigi Algorithm
MyDDSimpleMuonDigi.Lcio2EDM4hepTool = MyDDSimpleMuonDigiLCIOConv


MyRecoMCTruthLinker = MarlinProcessorWrapper("MyRecoMCTruthLinker")
MyRecoMCTruthLinker.OutputLevel = WARNING
MyRecoMCTruthLinker.ProcessorType = "RecoMCTruthLinker"
MyRecoMCTruthLinker.Parameters = {
    "BremsstrahlungEnergyCut": ["1"],
    "CalohitMCTruthLinkName": ["CalohitMCTruthLink"],
    "ClusterCollection": ["MergedClusters"],
    "ClusterMCTruthLinkName": ["ClusterMCTruthLink"],
    "FullRecoRelation": ["false"],
    "InvertedNonDestructiveInteractionLogic": ["false"],
    "KeepDaughtersPDG": ["22", "111", "310", "13", "211", "321", "3120"],
    "MCParticleCollection": ["MCPhysicsParticles"],
    "MCParticlesSkimmedName": ["MCParticlesSkimmed"],
    "MCTruthClusterLinkName": [],
    "MCTruthRecoLinkName": [],
    "MCTruthTrackLinkName": [],
    "RecoMCTruthLinkName": ["RecoMCTruthLink"],
    "RecoParticleCollection": ["MergedRecoParticles"],
    "SaveBremsstrahlungPhotons": ["false"],
    "SimCaloHitCollections": [
        "ECalBarrelCollection",
        "ECalEndcapCollection",
        "ECalPlugCollection",
        "HCalBarrelCollection",
        "HCalEndcapCollection",
        "HCalRingCollection",
        "YokeBarrelCollection",
        "YokeEndcapCollection",
        "LumiCalCollection",
        "BeamCalCollection",
    ],
    "SimCalorimeterHitRelationNames": ["RelationCaloHit", "RelationMuonHit"],
    "SimTrackerHitCollections": [
        "VertexBarrelCollection",
        "VertexEndcapCollection",
        "InnerTrackerBarrelCollection",
        "OuterTrackerBarrelCollection",
        "InnerTrackerEndcapCollection",
        "OuterTrackerEndcapCollection",
    ],
    "TrackCollection": ["SiTracks_Refitted"],
    "TrackMCTruthLinkName": ["SiTracksMCTruthLink"],
    "TrackerHitsRelInputCollections": [
        "VXDTrackerHitRelations",
        "VXDEndcapTrackerHitRelations",
        "InnerTrackerBarrelHitsRelations",
        "OuterTrackerBarrelHitsRelations",
        "InnerTrackerEndcapHitsRelations",
        "OuterTrackerEndcapHitsRelations",
    ],
    "UseTrackerHitRelations": ["true"],
    "UsingParticleGun": ["false"],
    "daughtersECutMeV": ["10"],
}
# LCIO to EDM4hep
MyRecoMCTruthLinkerLCIOConv = Lcio2EDM4hepTool("MyRecoMCTruthLinkerLCIOConv")
MyRecoMCTruthLinkerLCIOConv.convertAll = False
MyRecoMCTruthLinkerLCIOConv.collNameMapping = {
    "CalohitMCTruthLink": "CalohitMCTruthLink",
    "ClusterMCTruthLink": "ClusterMCTruthLink",
    "MCParticlesSkimmed": "MCParticlesSkimmed",
    "RecoMCTruthLink": "RecoMCTruthLink",
    "SiTracksMCTruthLink": "SiTracksMCTruthLink",
}
MyRecoMCTruthLinkerLCIOConv.OutputLevel = DEBUG
# Add it to MyRecoMCTruthLinker Algorithm
MyRecoMCTruthLinker.Lcio2EDM4hepTool = MyRecoMCTruthLinkerLCIOConv


MyHitResiduals = MarlinProcessorWrapper("MyHitResiduals")
MyHitResiduals.OutputLevel = WARNING
MyHitResiduals.ProcessorType = "HitResiduals"
MyHitResiduals.Parameters = {
    "EnergyLossOn": ["true"],
    "MaxChi2Increment": ["1000"],
    "MultipleScatteringOn": ["true"],
    "SmoothOn": ["false"],
    "TrackCollectionName": ["SiTracks_Refitted"],
    "outFileName": ["residuals.root"],
    "treeName": ["restree"],
}

LumiCalReco_Obs = MarlinProcessorWrapper("LumiCalReco_Obs")
LumiCalReco_Obs.OutputLevel = WARNING
LumiCalReco_Obs.ProcessorType = "MarlinLumiCalClusterer"
LumiCalReco_Obs.Parameters = {
    "ClusterMinNumHits": ["15"],
    "ElementsPercentInShowerPeakLayer": ["0.03"],
    "EnergyCalibConst": ["0.01213"],
    "LogWeigthConstant": ["6.5"],
    "LumiCal_Clusters": ["LumiCalClusters"],
    "LumiCal_Collection": ["LumiCalCollection"],
    "LumiCal_RecoParticles": ["LumiCalRecoParticles"],
    "MaxRecordNumber": ["5"],
    "MemoryResidentTree": ["0"],
    "MiddleEnergyHitBoundFrac": ["0.01"],
    "MinClusterEngy": ["2.0"],
    "MinHitEnergy": ["20e-06"],
    "MoliereRadius": ["20"],
    "NumEventsTree": ["500"],
    "NumOfNearNeighbor": ["6"],
    "OutDirName": ["rootOut"],
    "OutRootFileName": [],
    "SkipNEvents": ["0"],
    "WeightingMethod": ["LogMethod"],
    "ZLayerPhiOffset": ["0.0"],
}

LumiCalReco = MarlinProcessorWrapper("LumiCalReco")
LumiCalReco.OutputLevel = DEBUG
LumiCalReco.ProcessorType = "BeamCalClusterReco"
LumiCalReco.Parameters = {
    "BackgroundMethod": ["Empty"],
    "BeamCalCollectionName": ["LumiCalCollection"],
    "BeamCalHitsOutCollection": ["LumiCal_Hits"],
    "CreateEfficiencyFile": ["false"],
    "DetectorName": ["LumiCal"],
    "DetectorStartingLayerID": ["0"],
    "ETCluster": ["0.2"],
    "ETPad": ["20e-06"],
    "EfficiencyFilename": ["TaggingEfficiency.root"],
    "InputFileBackgrounds": [],
    "LinearCalibrationFactor": ["82.377"],
    "LogWeightingConstant": ["6.1"],
    "MCParticleCollectionName": ["MCParticle"],
    "MaxPadDistance": ["62"],
    "MinimumTowerSize": ["4"],
    "NShowerCountingLayers": ["30"],
    "NumberOfBX": ["0"],
    "PrintThisEvent": ["-1"],
    "RecoClusterCollectionname": ["LumiCalClusters"],
    "RecoParticleCollectionname": ["LumiCalRecoParticles"],
    "SigmaCut": ["1"],
    "StartLookingInLayer": ["1"],
    "StartingRing": ["0.0"],
    "SubClusterEnergyID": ["3"],
    "TowerChi2ndfLimit": ["5"],
    "UseChi2Selection": ["false"],
    "UseConstPadCuts": ["false"],
}
# LCIO to EDM4hep
LumiCalRecoLCIOConv = Lcio2EDM4hepTool("LumiCalRecoLCIOConv")
LumiCalRecoLCIOConv.convertAll = False
LumiCalRecoLCIOConv.collNameMapping = {
    "LumiCal_Hits": "LumiCal_Hits",
    "LumiCalClusters": "LumiCalClusters",
    "LumiCalRecoParticles": "LumiCalRecoParticles",
}
LumiCalRecoLCIOConv.OutputLevel = DEBUG
# Add it to LumiCalReco Algorithm
LumiCalReco.Lcio2EDM4hepTool = LumiCalRecoLCIOConv


RenameCollection = MarlinProcessorWrapper("RenameCollection")
RenameCollection.OutputLevel = WARNING
RenameCollection.ProcessorType = "MergeCollections"
RenameCollection.Parameters = {
    "CollectionParameterIndex": ["0"],
    "InputCollectionIDs": [],
    "InputCollections": ["PandoraPFOs"],
    "OutputCollection": ["PFOsFromJets"],
}
# LCIO to EDM4hep
RenameCollectionLCIOConv = Lcio2EDM4hepTool("RenameCollectionLCIOConv")
RenameCollectionLCIOConv.convertAll = False
RenameCollectionLCIOConv.collNameMapping = {
    "PFOsFromJets": "PFOsFromJets",
}
RenameCollectionLCIOConv.OutputLevel = DEBUG
# Add it to RenameCollection Algorithm
RenameCollection.Lcio2EDM4hepTool = RenameCollectionLCIOConv


MyFastJetProcessor = MarlinProcessorWrapper("MyFastJetProcessor")
MyFastJetProcessor.OutputLevel = WARNING
MyFastJetProcessor.ProcessorType = "FastJetProcessor"
MyFastJetProcessor.Parameters = {
    "algorithm": ["ValenciaPlugin", "1.2", "1.0", "0.7"],
    "clusteringMode": ["ExclusiveNJets", "2"],
    "jetOut": ["JetsAfterGamGamRemoval"],
    "recParticleIn": ["TightSelectedPandoraPFOs"],
    "recParticleOut": ["PFOsFromJets"],
    "recombinationScheme": ["E_scheme"],
    "storeParticlesInJets": ["true"],
}

OverlayFalse = MarlinProcessorWrapper("OverlayFalse")
OverlayFalse.OutputLevel = DEBUG
OverlayFalse.ProcessorType = "OverlayTimingGeneric"
OverlayFalse.Parameters = {
    "BackgroundFileNames": [],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["0"],
    "NumberBackground": ["0."],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}
# LCIO to EDM4hep
OverlayFalseLCIOConv = Lcio2EDM4hepTool("OverlayFalseLCIOConv")
OverlayFalseLCIOConv.convertAll = False
OverlayFalseLCIOConv.collNameMapping = {"MCPhysicsParticles": "MCPhysicsParticles"}
OverlayFalseLCIOConv.OutputLevel = DEBUG
# Add it to OverlayFalse Algorithm
OverlayFalse.Lcio2EDM4hepTool = OverlayFalseLCIOConv


Overlay350GeV_CDR = MarlinProcessorWrapper("Overlay350GeV_CDR")
Overlay350GeV_CDR.OutputLevel = WARNING
Overlay350GeV_CDR.ProcessorType = "OverlayTimingGeneric"
Overlay350GeV_CDR.Parameters = {
    "BackgroundFileNames": ["gghad_01.slcio", "gghad_02.slcio"],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["30"],
    "NumberBackground": ["0.0464"],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}

Overlay350GeV = MarlinProcessorWrapper("Overlay350GeV")
Overlay350GeV.OutputLevel = WARNING
Overlay350GeV.ProcessorType = "OverlayTimingGeneric"
Overlay350GeV.Parameters = {
    "BackgroundFileNames": ["gghad_01.slcio", "gghad_02.slcio"],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["30"],
    "NumberBackground": ["0.16"],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}

Overlay350GeV_L6 = MarlinProcessorWrapper("Overlay350GeV_L6")
Overlay350GeV_L6.OutputLevel = WARNING
Overlay350GeV_L6.ProcessorType = "OverlayTimingGeneric"
Overlay350GeV_L6.Parameters = {
    "BackgroundFileNames": ["gghad_01.slcio", "gghad_02.slcio"],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["30"],
    "NumberBackground": ["0.14"],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}

Overlay380GeV_CDR = MarlinProcessorWrapper("Overlay380GeV_CDR")
Overlay380GeV_CDR.OutputLevel = WARNING
Overlay380GeV_CDR.ProcessorType = "OverlayTimingGeneric"
Overlay380GeV_CDR.Parameters = {
    "BackgroundFileNames": ["gghad_01.slcio", "gghad_02.slcio"],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["30"],
    "NumberBackground": ["0.0464"],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}

Overlay380GeV = MarlinProcessorWrapper("Overlay380GeV")
Overlay380GeV.OutputLevel = WARNING
Overlay380GeV.ProcessorType = "OverlayTimingGeneric"
Overlay380GeV.Parameters = {
    "BackgroundFileNames": ["gghad_01.slcio", "gghad_02.slcio"],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["30"],
    "NumberBackground": ["0.18"],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}

Overlay380GeV_L6 = MarlinProcessorWrapper("Overlay380GeV_L6")
Overlay380GeV_L6.OutputLevel = WARNING
Overlay380GeV_L6.ProcessorType = "OverlayTimingGeneric"
Overlay380GeV_L6.Parameters = {
    "BackgroundFileNames": ["gghad_01.slcio", "gghad_02.slcio"],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["30"],
    "NumberBackground": ["0.178"],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}

Overlay420GeV = MarlinProcessorWrapper("Overlay420GeV")
Overlay420GeV.OutputLevel = WARNING
Overlay420GeV.ProcessorType = "OverlayTimingGeneric"
Overlay420GeV.Parameters = {
    "BackgroundFileNames": ["gghad_01.slcio", "gghad_02.slcio"],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["30"],
    "NumberBackground": ["0.17"],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}

Overlay500GeV = MarlinProcessorWrapper("Overlay500GeV")
Overlay500GeV.OutputLevel = WARNING
Overlay500GeV.ProcessorType = "OverlayTimingGeneric"
Overlay500GeV.Parameters = {
    "BackgroundFileNames": ["gghad_01.slcio", "gghad_02.slcio"],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["30"],
    "NumberBackground": ["0.3"],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}

Overlay1_4TeV = MarlinProcessorWrapper("Overlay1.4TeV")
Overlay1_4TeV.OutputLevel = WARNING
Overlay1_4TeV.ProcessorType = "OverlayTimingGeneric"
Overlay1_4TeV.Parameters = {
    "BackgroundFileNames": ["gghad_01.slcio", "gghad_02.slcio"],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["30"],
    "NumberBackground": ["1.3"],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}

Overlay3TeV = MarlinProcessorWrapper("Overlay3TeV")
Overlay3TeV.OutputLevel = WARNING
Overlay3TeV.ProcessorType = "OverlayTimingGeneric"
Overlay3TeV.Parameters = {
    "BackgroundFileNames": ["gghad_01.slcio", "gghad_02.slcio"],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["30"],
    "NumberBackground": ["3.2"],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}

Overlay3TeV_L6 = MarlinProcessorWrapper("Overlay3TeV_L6")
Overlay3TeV_L6.OutputLevel = WARNING
Overlay3TeV_L6.ProcessorType = "OverlayTimingGeneric"
Overlay3TeV_L6.Parameters = {
    "BackgroundFileNames": ["gghad_01.slcio", "gghad_02.slcio"],
    "Collection_IntegrationTimes": [
        "VertexBarrelCollection",
        "10",
        "VertexEndcapCollection",
        "10",
        "InnerTrackerBarrelCollection",
        "10",
        "InnerTrackerEndcapCollection",
        "10",
        "OuterTrackerBarrelCollection",
        "10",
        "OuterTrackerEndcapCollection",
        "10",
        "ECalBarrelCollection",
        "10",
        "ECalEndcapCollection",
        "10",
        "ECalPlugCollection",
        "10",
        "HCalBarrelCollection",
        "10",
        "HCalEndcapCollection",
        "10",
        "HCalRingCollection",
        "10",
        "YokeBarrelCollection",
        "10",
        "YokeEndcapCollection",
        "10",
        "LumiCalCollection",
        "10",
        "BeamCalCollection",
        "10",
    ],
    "Delta_t": ["0.5"],
    "MCParticleCollectionName": ["MCParticle"],
    "MCPhysicsParticleCollectionName": ["MCPhysicsParticles"],
    "NBunchtrain": ["30"],
    "NumberBackground": ["3.12"],
    "PhysicsBX": ["10"],
    "Poisson_random_NOverlay": ["true"],
    "RandomBx": ["false"],
    "TPCDriftvelocity": ["0.05"],
}

MergeRP = MarlinProcessorWrapper("MergeRP")
MergeRP.OutputLevel = WARNING
MergeRP.ProcessorType = "MergeCollections"
MergeRP.Parameters = {
    "CollectionParameterIndex": ["0"],
    "InputCollectionIDs": [],
    "InputCollections": ["PandoraPFOs", "LumiCalRecoParticles", "BeamCalRecoParticles"],
    "OutputCollection": ["MergedRecoParticles"],
}
# LCIO to EDM4hep
MergeRPLCIOConv = Lcio2EDM4hepTool("MergeRPLCIOConv")
MergeRPLCIOConv.convertAll = False
MergeRPLCIOConv.collNameMapping = {"MergedRecoParticles": "MergedRecoParticles"}
MergeRPLCIOConv.OutputLevel = DEBUG
# Add it to MergeRP Algorithm
MergeRP.Lcio2EDM4hepTool = MergeRPLCIOConv


MergeClusters = MarlinProcessorWrapper("MergeClusters")
MergeClusters.OutputLevel = WARNING
MergeClusters.ProcessorType = "MergeCollections"
MergeClusters.Parameters = {
    "CollectionParameterIndex": ["0"],
    "InputCollectionIDs": [],
    "InputCollections": ["PandoraClusters", "LumiCalClusterer", "BeamCalClusters"],
    "OutputCollection": ["MergedClusters"],
}
# LCIO to EDM4hep
MergeClustersLCIOConv = Lcio2EDM4hepTool("MergeClustersLCIOConv")
MergeClustersLCIOConv.convertAll = False
MergeClustersLCIOConv.collNameMapping = {"MergedClusters": "MergedClusters"}
MergeClustersLCIOConv.OutputLevel = DEBUG
# Add it to MergeClusters Algorithm
MergeClusters.Lcio2EDM4hepTool = MergeClustersLCIOConv


BeamCalReco3TeV = MarlinProcessorWrapper("BeamCalReco3TeV")
BeamCalReco3TeV.OutputLevel = WARNING
BeamCalReco3TeV.ProcessorType = "BeamCalClusterReco"
BeamCalReco3TeV.Parameters = {
    "BackgroundMethod": ["Gaussian"],
    "BeamCalCollectionName": ["BeamCalCollection"],
    "CreateEfficiencyFile": ["false"],
    "DetectorName": ["BeamCal"],
    "DetectorStartingLayerID": ["1"],
    "ETCluster": ["5.0", "4.0", "3.0", "2.0", "2.0", "1.0"],
    "ETPad": ["0.5", "0.4", "0.3", "0.2", "0.15", "0.1"],
    "EfficiencyFilename": ["TaggingEfficiency.root"],
    "InputFileBackgrounds": ["BeamCal_BackgroundPars_3TeV.root"],
    "LinearCalibrationFactor": ["116.44"],
    "MCParticleCollectionName": ["MCParticle"],
    "MinimumTowerSize": ["4"],
    "NShowerCountingLayers": ["3"],
    "NumberOfBX": ["40"],
    "PrintThisEvent": ["-1"],
    "RecoClusterCollectionname": ["BeamCalClusters"],
    "RecoParticleCollectionname": ["BeamCalRecoParticles"],
    "SigmaCut": ["1"],
    "StartLookingInLayer": ["10"],
    "StartingRing": ["0.0", "1.0", "1.5", "2.5", "3.5", "4.5"],
    "SubClusterEnergyID": ["5"],
    "TowerChi2ndfLimit": ["5"],
    "UseChi2Selection": ["false"],
    "UseConstPadCuts": ["true"],
}

BeamCalReco380GeV = MarlinProcessorWrapper("BeamCalReco380GeV")
BeamCalReco380GeV.OutputLevel = WARNING
BeamCalReco380GeV.ProcessorType = "BeamCalClusterReco"
BeamCalReco380GeV.Parameters = {
    "BackgroundMethod": ["Gaussian"],
    "BeamCalCollectionName": ["BeamCalCollection"],
    "CreateEfficiencyFile": ["false"],
    "DetectorName": ["BeamCal"],
    "DetectorStartingLayerID": ["1"],
    "ETCluster": ["1.0"],
    "ETPad": ["0.01"],
    "EfficiencyFilename": ["TaggingEfficiency.root"],
    "InputFileBackgrounds": ["BeamCal_BackgroundPars_380GeV.root"],
    "LinearCalibrationFactor": ["116.44"],
    "MCParticleCollectionName": ["MCParticle"],
    "MinimumTowerSize": ["4"],
    "NShowerCountingLayers": ["3"],
    "NumberOfBX": ["40"],
    "PrintThisEvent": ["-1"],
    "RecoClusterCollectionname": ["BeamCalClusters"],
    "RecoParticleCollectionname": ["BeamCalRecoParticles"],
    "SigmaCut": ["1"],
    "StartLookingInLayer": ["2"],
    "StartingRing": ["0.0"],
    "SubClusterEnergyID": ["5"],
    "TowerChi2ndfLimit": ["5"],
    "UseChi2Selection": ["false"],
    "UseConstPadCuts": ["false"],
}

CLICPfoSelectorDefault_HE = MarlinProcessorWrapper("CLICPfoSelectorDefault_HE")
CLICPfoSelectorDefault_HE.OutputLevel = WARNING
CLICPfoSelectorDefault_HE.ProcessorType = "CLICPfoSelector"
CLICPfoSelectorDefault_HE.Parameters = {
    "ChargedPfoLooseTimingCut": ["3"],
    "ChargedPfoNegativeLooseTimingCut": ["-1"],
    "ChargedPfoNegativeTightTimingCut": ["-0.5"],
    "ChargedPfoPtCut": ["0"],
    "ChargedPfoPtCutForLooseTiming": ["4"],
    "ChargedPfoTightTimingCut": ["1.5"],
    "CheckKaonCorrection": ["0"],
    "CheckProtonCorrection": ["0"],
    "ClusterLessPfoTrackTimeCut": ["10"],
    "CorrectHitTimesForTimeOfFlight": ["0"],
    "DisplayRejectedPfos": ["1"],
    "DisplaySelectedPfos": ["1"],
    "FarForwardCosTheta": ["0.975"],
    "ForwardCosThetaForHighEnergyNeutralHadrons": ["0.95"],
    "ForwardHighEnergyNeutralHadronsEnergy": ["10"],
    "HCalBarrelLooseTimingCut": ["20"],
    "HCalBarrelTightTimingCut": ["10"],
    "HCalEndCapTimingFactor": ["1"],
    "InputPfoCollection": ["PandoraPFOs"],
    "KeepKShorts": ["1"],
    "MaxMomentumForClusterLessPfos": ["2"],
    "MinECalHitsForTiming": ["5"],
    "MinHCalEndCapHitsForTiming": ["5"],
    "MinMomentumForClusterLessPfos": ["0.5"],
    "MinPtForClusterLessPfos": ["0.5"],
    "MinimumEnergyForNeutronTiming": ["1"],
    "Monitoring": ["0"],
    "MonitoringPfoEnergyToDisplay": ["1"],
    "NeutralFarForwardLooseTimingCut": ["2"],
    "NeutralFarForwardTightTimingCut": ["1"],
    "NeutralHadronBarrelPtCutForLooseTiming": ["3.5"],
    "NeutralHadronLooseTimingCut": ["2.5"],
    "NeutralHadronPtCut": ["0"],
    "NeutralHadronPtCutForLooseTiming": ["8"],
    "NeutralHadronTightTimingCut": ["1.5"],
    "PhotonFarForwardLooseTimingCut": ["2"],
    "PhotonFarForwardTightTimingCut": ["1"],
    "PhotonLooseTimingCut": ["2"],
    "PhotonPtCut": ["0"],
    "PhotonPtCutForLooseTiming": ["4"],
    "PhotonTightTimingCut": ["1"],
    "PtCutForTightTiming": ["0.75"],
    "SelectedPfoCollection": ["SelectedPandoraPFOs"],
    "UseClusterLessPfos": ["1"],
    "UseNeutronTiming": ["0"],
}
# LCIO to EDM4hep
CLICPfoSelectorDefault_HELCIOConv = Lcio2EDM4hepTool("CLICPfoSelectorDefault_HELCIOConv")
CLICPfoSelectorDefault_HELCIOConv.convertAll = False
CLICPfoSelectorDefault_HELCIOConv.collNameMapping = {"SelectedPandoraPFOs": "SelectedPandoraPFOs"}
CLICPfoSelectorDefault_HELCIOConv.OutputLevel = DEBUG
# Add it to CLICPfoSelectorDefault_HE Algorithm
CLICPfoSelectorDefault_HE.Lcio2EDM4hepTool = CLICPfoSelectorDefault_HELCIOConv


CLICPfoSelectorLoose_HE = MarlinProcessorWrapper("CLICPfoSelectorLoose_HE")
CLICPfoSelectorLoose_HE.OutputLevel = WARNING
CLICPfoSelectorLoose_HE.ProcessorType = "CLICPfoSelector"
CLICPfoSelectorLoose_HE.Parameters = {
    "ChargedPfoLooseTimingCut": ["3"],
    "ChargedPfoNegativeLooseTimingCut": ["-2.0"],
    "ChargedPfoNegativeTightTimingCut": ["-2.0"],
    "ChargedPfoPtCut": ["0"],
    "ChargedPfoPtCutForLooseTiming": ["4"],
    "ChargedPfoTightTimingCut": ["1.5"],
    "CheckKaonCorrection": ["0"],
    "CheckProtonCorrection": ["0"],
    "ClusterLessPfoTrackTimeCut": ["1000."],
    "CorrectHitTimesForTimeOfFlight": ["0"],
    "DisplayRejectedPfos": ["1"],
    "DisplaySelectedPfos": ["1"],
    "FarForwardCosTheta": ["0.975"],
    "ForwardCosThetaForHighEnergyNeutralHadrons": ["0.95"],
    "ForwardHighEnergyNeutralHadronsEnergy": ["10"],
    "HCalBarrelLooseTimingCut": ["20"],
    "HCalBarrelTightTimingCut": ["10"],
    "HCalEndCapTimingFactor": ["1"],
    "InputPfoCollection": ["PandoraPFOs"],
    "KeepKShorts": ["1"],
    "MaxMomentumForClusterLessPfos": ["2"],
    "MinECalHitsForTiming": ["5"],
    "MinHCalEndCapHitsForTiming": ["5"],
    "MinMomentumForClusterLessPfos": ["0.0"],
    "MinPtForClusterLessPfos": ["0.25"],
    "MinimumEnergyForNeutronTiming": ["1"],
    "Monitoring": ["0"],
    "MonitoringPfoEnergyToDisplay": ["1"],
    "NeutralFarForwardLooseTimingCut": ["2.5"],
    "NeutralFarForwardTightTimingCut": ["1.5"],
    "NeutralHadronBarrelPtCutForLooseTiming": ["3.5"],
    "NeutralHadronLooseTimingCut": ["2.5"],
    "NeutralHadronPtCut": ["0"],
    "NeutralHadronPtCutForLooseTiming": ["8"],
    "NeutralHadronTightTimingCut": ["1.5"],
    "PhotonFarForwardLooseTimingCut": ["2"],
    "PhotonFarForwardTightTimingCut": ["1"],
    "PhotonLooseTimingCut": ["2."],
    "PhotonPtCut": ["0"],
    "PhotonPtCutForLooseTiming": ["4"],
    "PhotonTightTimingCut": ["2."],
    "PtCutForTightTiming": ["0.75"],
    "SelectedPfoCollection": ["LooseSelectedPandoraPFOs"],
    "UseClusterLessPfos": ["1"],
    "UseNeutronTiming": ["0"],
}
# LCIO to EDM4hep
CLICPfoSelectorLoose_HELCIOConv = Lcio2EDM4hepTool("CLICPfoSelectorLoose_HELCIOConv")
CLICPfoSelectorLoose_HELCIOConv.convertAll = False
CLICPfoSelectorLoose_HELCIOConv.collNameMapping = {"CLICPfoSelectorLoose_HE": "CLICPfoSelectorLoose_HE"}
CLICPfoSelectorLoose_HELCIOConv.OutputLevel = DEBUG
# Add it to CLICPfoSelectorLoose_HE Algorithm
CLICPfoSelectorLoose_HE.Lcio2EDM4hepTool = CLICPfoSelectorLoose_HELCIOConv


CLICPfoSelectorTight_HE = MarlinProcessorWrapper("CLICPfoSelectorTight_HE")
CLICPfoSelectorTight_HE.OutputLevel = WARNING
CLICPfoSelectorTight_HE.ProcessorType = "CLICPfoSelector"
CLICPfoSelectorTight_HE.Parameters = {
    "ChargedPfoLooseTimingCut": ["2.0"],
    "ChargedPfoNegativeLooseTimingCut": ["-0.5"],
    "ChargedPfoNegativeTightTimingCut": ["-0.25"],
    "ChargedPfoPtCut": ["0"],
    "ChargedPfoPtCutForLooseTiming": ["4"],
    "ChargedPfoTightTimingCut": ["1.0"],
    "CheckKaonCorrection": ["0"],
    "CheckProtonCorrection": ["0"],
    "ClusterLessPfoTrackTimeCut": ["10"],
    "CorrectHitTimesForTimeOfFlight": ["0"],
    "DisplayRejectedPfos": ["1"],
    "DisplaySelectedPfos": ["1"],
    "FarForwardCosTheta": ["0.95"],
    "ForwardCosThetaForHighEnergyNeutralHadrons": ["0.95"],
    "ForwardHighEnergyNeutralHadronsEnergy": ["10"],
    "HCalBarrelLooseTimingCut": ["20"],
    "HCalBarrelTightTimingCut": ["10"],
    "HCalEndCapTimingFactor": ["1"],
    "InputPfoCollection": ["PandoraPFOs"],
    "KeepKShorts": ["1"],
    "MaxMomentumForClusterLessPfos": ["1.5"],
    "MinECalHitsForTiming": ["5"],
    "MinHCalEndCapHitsForTiming": ["5"],
    "MinMomentumForClusterLessPfos": ["0.5"],
    "MinPtForClusterLessPfos": ["1.0"],
    "MinimumEnergyForNeutronTiming": ["1"],
    "Monitoring": ["0"],
    "MonitoringPfoEnergyToDisplay": ["1"],
    "NeutralFarForwardLooseTimingCut": ["1.5"],
    "NeutralFarForwardTightTimingCut": ["1"],
    "NeutralHadronBarrelPtCutForLooseTiming": ["3.5"],
    "NeutralHadronLooseTimingCut": ["2.5"],
    "NeutralHadronPtCut": ["0.5"],
    "NeutralHadronPtCutForLooseTiming": ["8"],
    "NeutralHadronTightTimingCut": ["1.5"],
    "PhotonFarForwardLooseTimingCut": ["2"],
    "PhotonFarForwardTightTimingCut": ["1"],
    "PhotonLooseTimingCut": ["2"],
    "PhotonPtCut": ["0.2"],
    "PhotonPtCutForLooseTiming": ["4"],
    "PhotonTightTimingCut": ["1"],
    "PtCutForTightTiming": ["1.0"],
    "SelectedPfoCollection": ["TightSelectedPandoraPFOs"],
    "UseClusterLessPfos": ["0"],
    "UseNeutronTiming": ["0"],
}
# LCIO to EDM4hep
CLICPfoSelectorTight_HELCIOConv = Lcio2EDM4hepTool("CLICPfoSelectorTight_HELCIOConv")
CLICPfoSelectorTight_HELCIOConv.convertAll = False
CLICPfoSelectorTight_HELCIOConv.collNameMapping = {"TightSelectedPandoraPFOs": "TightSelectedPandoraPFOs"}
CLICPfoSelectorTight_HELCIOConv.OutputLevel = DEBUG
# Add it to CLICPfoSelectorTight_HE Algorithm
CLICPfoSelectorTight_HE.Lcio2EDM4hepTool = CLICPfoSelectorTight_HELCIOConv


CLICPfoSelectorDefault_LE = MarlinProcessorWrapper("CLICPfoSelectorDefault_LE")
CLICPfoSelectorDefault_LE.OutputLevel = WARNING
CLICPfoSelectorDefault_LE.ProcessorType = "CLICPfoSelector"
CLICPfoSelectorDefault_LE.Parameters = {
    "ChargedPfoLooseTimingCut": ["10.0"],
    "ChargedPfoNegativeLooseTimingCut": ["-5.0"],
    "ChargedPfoNegativeTightTimingCut": ["-2.0"],
    "ChargedPfoPtCut": ["0.0"],
    "ChargedPfoPtCutForLooseTiming": ["4.0"],
    "ChargedPfoTightTimingCut": ["3.0"],
    "CheckKaonCorrection": ["0"],
    "CheckProtonCorrection": ["0"],
    "ClusterLessPfoTrackTimeCut": ["10."],
    "CorrectHitTimesForTimeOfFlight": ["0"],
    "DisplayRejectedPfos": ["1"],
    "DisplaySelectedPfos": ["1"],
    "FarForwardCosTheta": ["0.975"],
    "ForwardCosThetaForHighEnergyNeutralHadrons": ["0.95"],
    "ForwardHighEnergyNeutralHadronsEnergy": ["10"],
    "HCalBarrelLooseTimingCut": ["5"],
    "HCalBarrelTightTimingCut": ["2.5"],
    "HCalEndCapTimingFactor": ["1"],
    "InputPfoCollection": ["PandoraPFOs"],
    "KeepKShorts": ["1"],
    "MaxMomentumForClusterLessPfos": ["5.0"],
    "MinECalHitsForTiming": ["5"],
    "MinHCalEndCapHitsForTiming": ["5"],
    "MinMomentumForClusterLessPfos": ["0.0"],
    "MinPtForClusterLessPfos": ["0.0"],
    "MinimumEnergyForNeutronTiming": ["1"],
    "Monitoring": ["0"],
    "MonitoringPfoEnergyToDisplay": ["1"],
    "NeutralFarForwardLooseTimingCut": ["4.0"],
    "NeutralFarForwardTightTimingCut": ["2.0"],
    "NeutralHadronBarrelPtCutForLooseTiming": ["3.5"],
    "NeutralHadronLooseTimingCut": ["5.0"],
    "NeutralHadronPtCut": ["0.0"],
    "NeutralHadronPtCutForLooseTiming": ["2.0"],
    "NeutralHadronTightTimingCut": ["2.5"],
    "PhotonFarForwardLooseTimingCut": ["2"],
    "PhotonFarForwardTightTimingCut": ["1"],
    "PhotonLooseTimingCut": ["5.0"],
    "PhotonPtCut": ["0.0"],
    "PhotonPtCutForLooseTiming": ["2.0"],
    "PhotonTightTimingCut": ["1.0"],
    "PtCutForTightTiming": ["0.75"],
    "SelectedPfoCollection": ["LE_SelectedPandoraPFOs"],
    "UseClusterLessPfos": ["1"],
    "UseNeutronTiming": ["0"],
}
# LCIO to EDM4hep
CLICPfoSelectorDefault_LELCIOConv = Lcio2EDM4hepTool("CLICPfoSelectorDefault_LELCIOConv")
CLICPfoSelectorDefault_LELCIOConv.convertAll = False
CLICPfoSelectorDefault_LELCIOConv.collNameMapping = {"LE_SelectedPandoraPFOs": "LE_SelectedPandoraPFOs"}
CLICPfoSelectorDefault_LELCIOConv.OutputLevel = DEBUG
# Add it to CLICPfoSelectorDefault_LE Algorithm
CLICPfoSelectorDefault_LE.Lcio2EDM4hepTool = CLICPfoSelectorDefault_LELCIOConv


CLICPfoSelectorLoose_LE = MarlinProcessorWrapper("CLICPfoSelectorLoose_LE")
CLICPfoSelectorLoose_LE.OutputLevel = WARNING
CLICPfoSelectorLoose_LE.ProcessorType = "CLICPfoSelector"
CLICPfoSelectorLoose_LE.Parameters = {
    "ChargedPfoLooseTimingCut": ["10.0"],
    "ChargedPfoNegativeLooseTimingCut": ["-20.0"],
    "ChargedPfoNegativeTightTimingCut": ["-20.0"],
    "ChargedPfoPtCut": ["0.0"],
    "ChargedPfoPtCutForLooseTiming": ["4.0"],
    "ChargedPfoTightTimingCut": ["5.0"],
    "CheckKaonCorrection": ["0"],
    "CheckProtonCorrection": ["0"],
    "ClusterLessPfoTrackTimeCut": ["50."],
    "CorrectHitTimesForTimeOfFlight": ["0"],
    "DisplayRejectedPfos": ["1"],
    "DisplaySelectedPfos": ["1"],
    "FarForwardCosTheta": ["0.975"],
    "ForwardCosThetaForHighEnergyNeutralHadrons": ["0.95"],
    "ForwardHighEnergyNeutralHadronsEnergy": ["10"],
    "HCalBarrelLooseTimingCut": ["10"],
    "HCalBarrelTightTimingCut": ["5"],
    "HCalEndCapTimingFactor": ["1"],
    "InputPfoCollection": ["PandoraPFOs"],
    "KeepKShorts": ["1"],
    "MaxMomentumForClusterLessPfos": ["5.0"],
    "MinECalHitsForTiming": ["5"],
    "MinHCalEndCapHitsForTiming": ["5"],
    "MinMomentumForClusterLessPfos": ["0.0"],
    "MinPtForClusterLessPfos": ["0.0"],
    "MinimumEnergyForNeutronTiming": ["1"],
    "Monitoring": ["0"],
    "MonitoringPfoEnergyToDisplay": ["1"],
    "NeutralFarForwardLooseTimingCut": ["10.0"],
    "NeutralFarForwardTightTimingCut": ["5.0"],
    "NeutralHadronBarrelPtCutForLooseTiming": ["3.5"],
    "NeutralHadronLooseTimingCut": ["10.0"],
    "NeutralHadronPtCut": ["0.0"],
    "NeutralHadronPtCutForLooseTiming": ["2.0"],
    "NeutralHadronTightTimingCut": ["5.0"],
    "PhotonFarForwardLooseTimingCut": ["2"],
    "PhotonFarForwardTightTimingCut": ["1"],
    "PhotonLooseTimingCut": ["10.0"],
    "PhotonPtCut": ["0.0"],
    "PhotonPtCutForLooseTiming": ["2.0"],
    "PhotonTightTimingCut": ["2.5"],
    "PtCutForTightTiming": ["0.75"],
    "SelectedPfoCollection": ["LE_LooseSelectedPandoraPFOs"],
    "UseClusterLessPfos": ["1"],
    "UseNeutronTiming": ["0"],
}
# LCIO to EDM4hep
CLICPfoSelectorLoose_LELCIOConv = Lcio2EDM4hepTool("CLICPfoSelectorLoose_LELCIOConv")
CLICPfoSelectorLoose_LELCIOConv.convertAll = False
CLICPfoSelectorLoose_LELCIOConv.collNameMapping = {"LE_LooseSelectedPandoraPFOs": "LE_LooseSelectedPandoraPFOs"}
CLICPfoSelectorLoose_LELCIOConv.OutputLevel = DEBUG
# Add it to CLICPfoSelectorLoose_LE Algorithm
CLICPfoSelectorLoose_LE.Lcio2EDM4hepTool = CLICPfoSelectorLoose_LELCIOConv


CLICPfoSelectorTight_LE = MarlinProcessorWrapper("CLICPfoSelectorTight_LE")
CLICPfoSelectorTight_LE.OutputLevel = WARNING
CLICPfoSelectorTight_LE.ProcessorType = "CLICPfoSelector"
CLICPfoSelectorTight_LE.Parameters = {
    "ChargedPfoLooseTimingCut": ["4.0"],
    "ChargedPfoNegativeLooseTimingCut": ["-2.0"],
    "ChargedPfoNegativeTightTimingCut": ["-1.0"],
    "ChargedPfoPtCut": ["0.0"],
    "ChargedPfoPtCutForLooseTiming": ["3.0"],
    "ChargedPfoTightTimingCut": ["2.0"],
    "CheckKaonCorrection": ["0"],
    "CheckProtonCorrection": ["0"],
    "ClusterLessPfoTrackTimeCut": ["10."],
    "CorrectHitTimesForTimeOfFlight": ["0"],
    "DisplayRejectedPfos": ["1"],
    "DisplaySelectedPfos": ["1"],
    "FarForwardCosTheta": ["0.975"],
    "ForwardCosThetaForHighEnergyNeutralHadrons": ["0.95"],
    "ForwardHighEnergyNeutralHadronsEnergy": ["10"],
    "HCalBarrelLooseTimingCut": ["4"],
    "HCalBarrelTightTimingCut": ["2"],
    "HCalEndCapTimingFactor": ["1"],
    "InputPfoCollection": ["PandoraPFOs"],
    "KeepKShorts": ["1"],
    "MaxMomentumForClusterLessPfos": ["5.0"],
    "MinECalHitsForTiming": ["5"],
    "MinHCalEndCapHitsForTiming": ["5"],
    "MinMomentumForClusterLessPfos": ["0.0"],
    "MinPtForClusterLessPfos": ["0.75"],
    "MinimumEnergyForNeutronTiming": ["1"],
    "Monitoring": ["0"],
    "MonitoringPfoEnergyToDisplay": ["1"],
    "NeutralFarForwardLooseTimingCut": ["2.0"],
    "NeutralFarForwardTightTimingCut": ["2.0"],
    "NeutralHadronBarrelPtCutForLooseTiming": ["3.5"],
    "NeutralHadronLooseTimingCut": ["4.0"],
    "NeutralHadronPtCut": ["0.0"],
    "NeutralHadronPtCutForLooseTiming": ["3.0"],
    "NeutralHadronTightTimingCut": ["2.0"],
    "PhotonFarForwardLooseTimingCut": ["2"],
    "PhotonFarForwardTightTimingCut": ["1"],
    "PhotonLooseTimingCut": ["1.0"],
    "PhotonPtCut": ["0.0"],
    "PhotonPtCutForLooseTiming": ["2.0"],
    "PhotonTightTimingCut": ["1.0"],
    "PtCutForTightTiming": ["0.75"],
    "SelectedPfoCollection": ["LE_TightSelectedPandoraPFOs"],
    "UseClusterLessPfos": ["1"],
    "UseNeutronTiming": ["0"],
}
# LCIO to EDM4hep
CLICPfoSelectorTight_LELCIOConv = Lcio2EDM4hepTool("CLICPfoSelectorTight_LELCIOConv")
CLICPfoSelectorTight_LELCIOConv.convertAll = False
CLICPfoSelectorTight_LELCIOConv.collNameMapping = {"LE_TightSelectedPandoraPFOs": "LE_TightSelectedPandoraPFOs"}
CLICPfoSelectorTight_LELCIOConv.OutputLevel = DEBUG
# Add it to CLICPfoSelectorTight_LE Algorithm
CLICPfoSelectorTight_LE.Lcio2EDM4hepTool = CLICPfoSelectorTight_LELCIOConv


VertexFinder = MarlinProcessorWrapper("VertexFinder")
VertexFinder.OutputLevel = WARNING
VertexFinder.ProcessorType = "LcfiplusProcessor"
VertexFinder.Parameters = {
    "Algorithms": ["PrimaryVertexFinder", "BuildUpVertex"],
    "BeamSizeX": ["40.E-6"],
    "BeamSizeY": ["1.0E-6"],
    "BeamSizeZ": ["44E-3"],
    "BuildUpVertex.AVFTemperature": ["5.0"],
    "BuildUpVertex.AssocIPTracks": ["1"],
    "BuildUpVertex.AssocIPTracksChi2RatioSecToPri": ["2.0"],
    "BuildUpVertex.AssocIPTracksMinDist": ["0."],
    "BuildUpVertex.MassThreshold": ["10."],
    "BuildUpVertex.MaxChi2ForDistOrder": ["1.0"],
    "BuildUpVertex.MinDistFromIP": ["0.3"],
    "BuildUpVertex.PrimaryChi2Threshold": ["25."],
    "BuildUpVertex.SecondaryChi2Threshold": ["9."],
    "BuildUpVertex.TrackMaxD0": ["10."],
    "BuildUpVertex.TrackMaxD0Err": ["0.1"],
    "BuildUpVertex.TrackMaxZ0": ["20."],
    "BuildUpVertex.TrackMaxZ0Err": ["0.1"],
    "BuildUpVertex.TrackMinFtdHits": ["1"],
    "BuildUpVertex.TrackMinPt": ["0.1"],
    "BuildUpVertex.TrackMinTpcHits": ["1"],
    "BuildUpVertex.TrackMinTpcHitsMinPt": ["999999"],
    "BuildUpVertex.TrackMinVxdFtdHits": ["1"],
    "BuildUpVertex.TrackMinVxdHits": ["1"],
    "BuildUpVertex.UseAVF": ["false"],
    "BuildUpVertex.UseV0Selection": ["1"],
    "BuildUpVertex.V0VertexCollectionName": ["BuildUpVertices_V0"],
    "BuildUpVertexCollectionName": ["BuildUpVertices"],
    "MCPCollection": ["MCParticle"],
    "MCPFORelation": ["RecoMCTruthLink"],
    "MagneticField": ["4.0"],
    "PFOCollection": ["PFOsFromJets"],
    "PrimaryVertexCollectionName": ["PrimaryVertices"],
    "PrimaryVertexFinder.BeamspotConstraint": ["1"],
    "PrimaryVertexFinder.BeamspotSmearing": ["false"],
    "PrimaryVertexFinder.Chi2Threshold": ["25."],
    "PrimaryVertexFinder.TrackMaxD0": ["20."],
    "PrimaryVertexFinder.TrackMaxInnermostHitRadius": ["61"],
    "PrimaryVertexFinder.TrackMaxZ0": ["20."],
    "PrimaryVertexFinder.TrackMinFtdHits": ["999999"],
    "PrimaryVertexFinder.TrackMinTpcHits": ["999999"],
    "PrimaryVertexFinder.TrackMinTpcHitsMinPt": ["999999"],
    "PrimaryVertexFinder.TrackMinVtxFtdHits": ["1"],
    "PrimaryVertexFinder.TrackMinVxdHits": ["999999"],
    "PrintEventNumber": ["1"],
    "ReadSubdetectorEnergies": ["0"],
    "TrackHitOrdering": ["2"],
    "UpdateVertexRPDaughters": ["0"],
    "UseMCP": ["0"],
}
# LCIO to EDM4hep
VertexFinderLCIOConv = Lcio2EDM4hepTool("VertexFinderLCIOConv")
VertexFinderLCIOConv.convertAll = False
VertexFinderLCIOConv.collNameMapping = {
    "BuildUpVertices_V0": "BuildUpVertices_V0",
    "BuildUpVertices": "BuildUpVertices",
    "PrimaryVertices": "PrimaryVertices",
}
VertexFinderLCIOConv.OutputLevel = DEBUG
# Add it to VertexFinder Algorithm
VertexFinder.Lcio2EDM4hepTool = VertexFinderLCIOConv


VertexFinderUnconstrained = MarlinProcessorWrapper("VertexFinderUnconstrained")
VertexFinderUnconstrained.OutputLevel = WARNING
VertexFinderUnconstrained.ProcessorType = "LcfiplusProcessor"
VertexFinderUnconstrained.Parameters = {
    "Algorithms": ["PrimaryVertexFinder", "BuildUpVertex"],
    "BeamSizeX": ["40.E-6"],
    "BeamSizeY": ["1.0E-6"],
    "BeamSizeZ": ["44E-3"],
    "BuildUpVertex.AVFTemperature": ["5.0"],
    "BuildUpVertex.AssocIPTracks": ["1"],
    "BuildUpVertex.AssocIPTracksChi2RatioSecToPri": ["2.0"],
    "BuildUpVertex.AssocIPTracksMinDist": ["0."],
    "BuildUpVertex.MassThreshold": ["10."],
    "BuildUpVertex.MaxChi2ForDistOrder": ["1.0"],
    "BuildUpVertex.MinDistFromIP": ["0.3"],
    "BuildUpVertex.PrimaryChi2Threshold": ["25."],
    "BuildUpVertex.SecondaryChi2Threshold": ["9."],
    "BuildUpVertex.TrackMaxD0": ["10."],
    "BuildUpVertex.TrackMaxD0Err": ["0.1"],
    "BuildUpVertex.TrackMaxZ0": ["20."],
    "BuildUpVertex.TrackMaxZ0Err": ["0.1"],
    "BuildUpVertex.TrackMinFtdHits": ["1"],
    "BuildUpVertex.TrackMinPt": ["0.1"],
    "BuildUpVertex.TrackMinTpcHits": ["1"],
    "BuildUpVertex.TrackMinTpcHitsMinPt": ["999999"],
    "BuildUpVertex.TrackMinVxdFtdHits": ["1"],
    "BuildUpVertex.TrackMinVxdHits": ["1"],
    "BuildUpVertex.UseAVF": ["false"],
    "BuildUpVertex.UseV0Selection": ["1"],
    "BuildUpVertex.V0VertexCollectionName": ["BuildUpVertices_V0_res"],
    "BuildUpVertexCollectionName": ["BuildUpVertices_res"],
    "MCPCollection": ["MCParticle"],
    "MCPFORelation": ["RecoMCTruthLink"],
    "MagneticField": ["4.0"],
    "PFOCollection": ["TightSelectedPandoraPFOs"],
    "PrimaryVertexCollectionName": ["PrimaryVertices_res"],
    "PrimaryVertexFinder.BeamspotConstraint": ["0"],
    "PrimaryVertexFinder.BeamspotSmearing": ["false"],
    "PrimaryVertexFinder.Chi2Threshold": ["25."],
    "PrimaryVertexFinder.TrackMaxD0": ["20."],
    "PrimaryVertexFinder.TrackMaxInnermostHitRadius": ["61"],
    "PrimaryVertexFinder.TrackMaxZ0": ["20."],
    "PrimaryVertexFinder.TrackMinFtdHits": ["999999"],
    "PrimaryVertexFinder.TrackMinTpcHits": ["999999"],
    "PrimaryVertexFinder.TrackMinTpcHitsMinPt": ["999999"],
    "PrimaryVertexFinder.TrackMinVtxFtdHits": ["1"],
    "PrimaryVertexFinder.TrackMinVxdHits": ["999999"],
    "PrintEventNumber": ["1"],
    "ReadSubdetectorEnergies": ["0"],
    "TrackHitOrdering": ["2"],
    "UpdateVertexRPDaughters": ["0"],
    "UseMCP": ["0"],
}


# Write output to EDM4hep
from Configurables import PodioOutput

out = PodioOutput("PodioOutput", filename="my_output.root")
out.outputCommands = ["keep *"]
out.OutputLevel = INFO
out.outputCommands = [
    "drop *",
    "keep MCParticles*",
    "keep Pandora*",
    "keep SiTracks*",
    "keep *MCTruthLink*",
    "keep *Clusters*",
    "keep *RecoParticles*",
    "keep *MUON*",
    "keep *ECALBarrel*",
    "keep *ECALEndcap*",
    "keep *ECALOther*",
    "keep *HCALBarrel*",
    "keep *HCALEndcap*",
    "keep *HCALOther*",
    "keep *TrackerHit*",
    "keep *Vertices*",
    "drop LumiCal*",
]


algList.append(inp)
algList.append(MyAIDAProcessor)
algList.append(EventNumber)
algList.append(InitDD4hep)
algList.append(Config)
algList.append(OverlayFalse)  # Config.OverlayFalse
# algList.append(Overlay350GeV_CDR)  # Config.Overlay350GeV_CDR
# algList.append(Overlay350GeV)  # Config.Overlay350GeV
# algList.append(Overlay350GeV_L6)  # Config.Overlay350GeV_L6
# algList.append(Overlay380GeV_CDR)  # Config.Overlay380GeV_CDR
# algList.append(Overlay380GeV)  # Config.Overlay380GeV
# algList.append(Overlay380GeV_L6)  # Config.Overlay380GeV_L6
# algList.append(Overlay500GeV)  # Config.Overlay500GeV
# algList.append(Overlay1_4TeV)  # Config.Overlay1.4TeV
# algList.append(Overlay3TeV)  # Config.Overlay3TeV
# algList.append(Overlay3TeV_L6)  # Config.Overlay3TeV_L6
algList.append(VXDBarrelDigitiser)
algList.append(VXDEndcapDigitiser)
algList.append(InnerPlanarDigiProcessor)
algList.append(InnerEndcapPlanarDigiProcessor)
algList.append(OuterPlanarDigiProcessor)
algList.append(OuterEndcapPlanarDigiProcessor)
# algList.append(MyTruthTrackFinder)  # Config.TrackingTruth
algList.append(MyConformalTracking)  # Config.TrackingConformal
algList.append(ClonesAndSplitTracksFinder)  # Config.TrackingConformal
algList.append(Refit)
algList.append(MyDDCaloDigi)
algList.append(MyDDSimpleMuonDigi)
algList.append(MyDDMarlinPandora)
algList.append(LumiCalReco)
# # algList.append(BeamCalReco3TeV)  # Config.BeamCal3TeV
# # algList.append(BeamCalReco380GeV)  # Config.BeamCal380GeV
algList.append(MergeRP)
algList.append(MergeClusters)
algList.append(MyClicEfficiencyCalculator)
algList.append(MyRecoMCTruthLinker)
algList.append(MyTrackChecker)
algList.append(CLICPfoSelectorDefault_HE)
algList.append(CLICPfoSelectorLoose_HE)
algList.append(CLICPfoSelectorTight_HE)
algList.append(CLICPfoSelectorDefault_LE)
algList.append(CLICPfoSelectorLoose_LE)
algList.append(CLICPfoSelectorTight_LE)
algList.append(RenameCollection)  # Config.OverlayFalse
# # algList.append(MyFastJetProcessor)  # Config.OverlayNotFalse
algList.append(VertexFinder)
# algList.append(JetClusteringAndRefiner)
# # algList.append(VertexFinderUnconstrained)  # Config.VertexUnconstrainedON
# algList.append(Output_REC)
# algList.append(Output_DST)
algList.append(out)

from Configurables import ApplicationMgr

ApplicationMgr(TopAlg=algList, EvtSel="NONE", EvtMax=3, ExtSvc=[evtsvc], OutputLevel=WARNING)
