! CLD ECAL granularity scan: Z->tau tau at 91 GeV. The beam spot below
! matches CLDReconstruction.py's 91 GeV condition; all pitch points use it.
! adapted from https://github.com/HEP-FCC/FCC-config/blob/winter2023/FCCee/Generator/Pythia8/p8_ee_ZZ_ecm365.cmd
! Apache License 2.0, copyright FCC-config authors, modified by Joosep Pata in 2025

Random:setSeed = on
Main:numberOfEvents = 1000         ! number of events to generate
Main:timesAllowErrors = 5          ! how many aborts before run stops

! 2) Settings related to output in init(), next() and stat().
Init:showChangedSettings = on      ! list changed settings
Init:showChangedParticleData = off ! list changed particle data
Next:numberCount = 100             ! print message every n events
Next:numberShowInfo = 1            ! print event information n times
Next:numberShowProcess = 1         ! print process record n times
Next:numberShowEvent = 0           ! print event record n times

Beams:idA = 11                   ! first beam, e+ = 11
Beams:idB = -11                   ! second beam, e- = -11

! Vertex smearing :
Beams:allowVertexSpread = on
Beams:sigmaVertexX = 5.96e-3   !  5.96 mum
Beams:sigmaVertexY = 23.8e-6   !  23.8 nm
Beams:sigmaVertexZ = 0.397     !  0.397 mm

! 3) Hard process : Z production at 91GeV
Beams:eCM = 91  ! CM energy of collision
WeakSingleBoson:ffbar2ffbar(s:gmZ) = on

! 4) Settings for the event generation process in the Pythia8 library.
PartonLevel:ISR = on               ! initial-state radiation
PartonLevel:FSR = on               ! final-state radiation

! 5) Non-standard settings; exemplifies tuning possibilities.
23:onMode   = off
23:onIfAny  = 15
