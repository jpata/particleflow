# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO --runUnscheduled --conditions auto:phase1_2021_realistic -s RAW2DIGI,L1Reco,RECO,RECOSIM,EI,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM --eventcontent RECOSIM,MINIAODSIM,DQM -n 100 --filein file:step2.root --fileout file:step3.root --no_exec --era Run3 --scenario pp --geometry DB:Extended --mc
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process("RECO",Run3)

# import of standard configurations
process.load("Configuration.StandardSequences.Services_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimGeneral.MixingModule.mix_Run3_Flat55To75_PoissonOOTPU_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.L1Reco_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.RecoSim_cff")
process.load("CommonTools.ParticleFlow.EITopPAG_cff")
process.load("PhysicsTools.PatAlgos.slimming.metFilterPaths_cff")
process.load("Configuration.StandardSequences.PATMC_cff")
#process.load("Configuration.StandardSequences.Validation_cff")
process.load("DQMServices.Core.DQMStoreNonLegacy_cff")
process.load("DQMOffline.Configuration.DQMOfflineMC_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cff")
#process.load("RecoParticleFlow.PFTracking.hgcalTrackCollection_cfi")
#process.load("SimTracker.TrackerHitAssociation.tpClusterProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.trackingTruthProducer_cfi")
process.load("RecoParticleFlow.PFProducer.simPFProducer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file://./output_numEvent10.root"),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(

        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string("FULLMERGE"),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string("step3 nevts:100"),
    name = cms.untracked.string("Applications"),
    version = cms.untracked.string("$Revision: 1.19 $")
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string("GEN-SIM-RECO"),
        filterName = cms.untracked.string("")
    ),
    fileName = cms.untracked.string("file:step3.root"),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.MINIAODSIMoutput = cms.OutputModule("PoolOutputModule",
    compressionAlgorithm = cms.untracked.string("LZMA"),
    compressionLevel = cms.untracked.int32(4),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string("MINIAODSIM"),
        filterName = cms.untracked.string("")
    ),
    dropMetaData = cms.untracked.string("ALL"),
    eventAutoFlushCompressedSize = cms.untracked.int32(-900),
    fastCloning = cms.untracked.bool(False),
    fileName = cms.untracked.string("file:step3_inMINIAODSIM.root"),
    outputCommands = process.MINIAODSIMEventContent.outputCommands,
    overrideBranchesSplitLevel = cms.untracked.VPSet(
        cms.untracked.PSet(
            branch = cms.untracked.string("patPackedCandidates_packedPFCandidates__*"),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string("recoGenParticles_prunedGenParticles__*"),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string("patTriggerObjectStandAlones_slimmedPatTrigger__*"),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string("patPackedGenParticles_packedGenParticles__*"),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string("patJets_slimmedJets__*"),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string("recoVertexs_offlineSlimmedPrimaryVertices__*"),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string("recoCaloClusters_reducedEgamma_reducedESClusters_*"),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string("EcalRecHitsSorted_reducedEgamma_reducedEBRecHits_*"),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string("EcalRecHitsSorted_reducedEgamma_reducedEERecHits_*"),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string("recoGenJets_slimmedGenJets__*"),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string("patJets_slimmedJetsPuppi__*"),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string("EcalRecHitsSorted_reducedEgamma_reducedESRecHits_*"),
            splitLevel = cms.untracked.int32(99)
        )
    ),
    overrideInputFileSplitLevels = cms.untracked.bool(True),
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string("DQMIO"),
        filterName = cms.untracked.string("")
    ),
    fileName = cms.untracked.string("file:step3_inDQM.root"),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.playback = True
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase1_2021_realistic", "")

process.genParticlePlusGeant = cms.EDProducer("GenPlusSimParticleProducer",
    src = cms.InputTag("g4SimHits"),
    setStatus = cms.int32(8),
    filter = cms.vstring("pt > 0.0"),
    genParticles = cms.InputTag("genParticles")
)

process.genSequence = cms.Sequence(
    process.mix*
    process.genParticlePlusGeant*
    process.particleFlowSimParticle
#    process.simHitTPAssocProducer*
#    process.trackAssociatorByHits
)

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.recosim_step = cms.Path(process.recosim)
process.eventinterpretaion_step = cms.Path(process.EIsequence)
process.Flag_trackingFailureFilter = cms.Path(process.goodVertices+process.trackingFailureFilter)
process.Flag_goodVertices = cms.Path(process.primaryVertexFilter)
process.Flag_CSCTightHaloFilter = cms.Path(process.CSCTightHaloFilter)
process.Flag_trkPOGFilters = cms.Path(process.trkPOGFilters)
process.Flag_HcalStripHaloFilter = cms.Path(process.HcalStripHaloFilter)
process.Flag_trkPOG_logErrorTooManyClusters = cms.Path(~process.logErrorTooManyClusters)
process.Flag_EcalDeadCellTriggerPrimitiveFilter = cms.Path(process.EcalDeadCellTriggerPrimitiveFilter)
process.Flag_ecalLaserCorrFilter = cms.Path(process.ecalLaserCorrFilter)
process.Flag_globalSuperTightHalo2016Filter = cms.Path(process.globalSuperTightHalo2016Filter)
process.Flag_eeBadScFilter = cms.Path(process.eeBadScFilter)
process.Flag_METFilters = cms.Path(process.metFilters)
process.Flag_chargedHadronTrackResolutionFilter = cms.Path(process.chargedHadronTrackResolutionFilter)
process.Flag_globalTightHalo2016Filter = cms.Path(process.globalTightHalo2016Filter)
process.Flag_CSCTightHaloTrkMuUnvetoFilter = cms.Path(process.CSCTightHaloTrkMuUnvetoFilter)
process.Flag_HBHENoiseIsoFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseIsoFilter)
process.Flag_BadChargedCandidateSummer16Filter = cms.Path(process.BadChargedCandidateSummer16Filter)
process.Flag_hcalLaserEventFilter = cms.Path(process.hcalLaserEventFilter)
process.Flag_BadPFMuonFilter = cms.Path(process.BadPFMuonFilter)
process.Flag_ecalBadCalibFilter = cms.Path(process.ecalBadCalibFilter)
process.Flag_HBHENoiseFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseFilter)
process.Flag_trkPOG_toomanystripclus53X = cms.Path(~process.toomanystripclus53X)
process.Flag_EcalDeadCellBoundaryEnergyFilter = cms.Path(process.EcalDeadCellBoundaryEnergyFilter)
process.Flag_BadChargedCandidateFilter = cms.Path(process.BadChargedCandidateFilter)
process.Flag_trkPOG_manystripclus53X = cms.Path(~process.manystripclus53X)
process.Flag_BadPFMuonSummer16Filter = cms.Path(process.BadPFMuonSummer16Filter)
process.Flag_muonBadTrackFilter = cms.Path(process.muonBadTrackFilter)
process.Flag_CSCTightHalo2015Filter = cms.Path(process.CSCTightHalo2015Filter)
#process.prevalidation_step = cms.Path(process.prevalidationNoHLT)
#process.prevalidation_step1 = cms.Path(process.prevalidationMiniAOD)

#process.validation_step = cms.EndPath(process.validationNoHLT)
#process.validation_step1 = cms.EndPath(process.validationMiniAOD)
process.dqmoffline_step = cms.EndPath(process.DQMOfflineFakeHLT)
process.dqmoffline_1_step = cms.EndPath(process.DQMOfflineMiniAOD)
process.dqmofflineOnPAT_step = cms.EndPath(process.PostDQMOffline)
process.dqmofflineOnPAT_1_step = cms.EndPath(process.PostDQMOfflineMiniAOD)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)
process.MINIAODSIMoutput_step = cms.EndPath(process.MINIAODSIMoutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.recosim_step,process.eventinterpretaion_step,process.Flag_HBHENoiseFilter,process.Flag_HBHENoiseIsoFilter,process.Flag_CSCTightHaloFilter,process.Flag_CSCTightHaloTrkMuUnvetoFilter,process.Flag_CSCTightHalo2015Filter,process.Flag_globalTightHalo2016Filter,process.Flag_globalSuperTightHalo2016Filter,process.Flag_HcalStripHaloFilter,process.Flag_hcalLaserEventFilter,process.Flag_EcalDeadCellTriggerPrimitiveFilter,process.Flag_EcalDeadCellBoundaryEnergyFilter,process.Flag_ecalBadCalibFilter,process.Flag_goodVertices,process.Flag_eeBadScFilter,process.Flag_ecalLaserCorrFilter,process.Flag_trkPOGFilters,process.Flag_chargedHadronTrackResolutionFilter,process.Flag_muonBadTrackFilter,process.Flag_BadChargedCandidateFilter,process.Flag_BadPFMuonFilter,process.Flag_BadChargedCandidateSummer16Filter,process.Flag_BadPFMuonSummer16Filter,process.Flag_trkPOG_manystripclus53X,process.Flag_trkPOG_toomanystripclus53X,process.Flag_trkPOG_logErrorTooManyClusters,process.Flag_METFilters,process.dqmoffline_step,process.dqmoffline_1_step,process.dqmofflineOnPAT_step,process.dqmofflineOnPAT_1_step,process.RECOSIMoutput_step,process.MINIAODSIMoutput_step,process.DQMoutput_step)
process.schedule.associate(process.patTask)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
#from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 
#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
#process = setCrossingFrameOn(process)

# End of customisation functions
#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC 

#call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllMC(process)

# End of customisation functions

# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
#from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
#process = customiseEarlyDelete(process)
# End adding early deletion

#To keep low-level inputs
extra_keeps = [
  "keep recoPFClusters_particleFlowClusterECAL_*_*",
  "keep recoPFClusters_particleFlowClusterHCAL_*_*",
  "keep recoPFClusters_particleFlowClusterHO_*_*",
  "keep recoPFClusters_particleFlowClusterHF_*_*",
  "keep recoPFClusters_particleFlowClusterPS_*_*",
  "keep recoTracks_*_*_*",
  "keep recoPFRecTracks_*_*_*",
  "keep recoGsfPFRecTracks_*_*_*",
  "keep recoTrackExtras_*_*_*",
  "keep TrackingRecHitsOwned_generalTracks_*_*",
  "keep recoGenParticles_prunedGenParticles_*_*",
  "keep recoPFBlocks_particleFlowBlock_*_*",
  "keep recoMuons_*_*_*",
  "keep *_pileupInformation_*_*",
  "keep *_g4SimHits_*_*",
  "keep *_genParticlePlusGeant_*_*",
  "keep *_particleFlowSimParticle_*_*",
  "keep *_simPFProducer_*_*",
  "keep *_mix_*_*",
  "keep *_trackAssociatorByHits_*_*",
  "keep *_generatorSmeared_*_*",
]
process.genPath = cms.Path(process.genSequence)
process.schedule.insert(0, process.genPath)

process.AODSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string("AODSIM"),
        filterName = cms.untracked.string("")
    ),
    fileName = cms.untracked.string("file:step3_AOD.root"),
    outputCommands = process.AODSIMEventContent.outputCommands + extra_keeps,
    splitLevel = cms.untracked.int32(0)
)
process.AODSIMoutput_step = cms.EndPath(process.AODSIMoutput)
process.schedule.insert(-1, process.AODSIMoutput_step)

#process.caloParticles.simHitCollections = cms.PSet(
#    hcal = cms.VInputTag(cms.InputTag('g4SimHits','HcalHits')),
#    ecal = cms.VInputTag(
#        cms.InputTag('g4SimHits','EcalHitsEE'),
#        cms.InputTag('g4SimHits','EcalHitsEB'),
#        cms.InputTag('g4SimHits','EcalHitsES')
#    )
#)
#process.trackingParticles.ignoreTracksOutsideVolume = True
#process.trackingParticles.alwaysAddAncestors = False
#seems like the reliable way to run digitizers is in step2
#process.mix.digitizers = cms.PSet(
#    #crashes as it's HGCAL-specific
#    #calotruth=cms.PSet(process.caloParticles),
#    mergedtruth=cms.PSet(process.trackingParticles)
#)
#process.MessageLogger.cerr.INFO = cms.untracked.PSet(limit = cms.untracked.int32(1000000))
