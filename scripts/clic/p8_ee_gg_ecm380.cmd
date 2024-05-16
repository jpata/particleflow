Random:setSeed = on
Main:numberOfEvents = 100         ! number of events to generate
Main:timesAllowErrors = 5          ! how many aborts before run stops

! 2) Settings related to output in init(), next() and stat().
Init:showChangedSettings = on      ! list changed settings
Init:showChangedParticleData = off ! list changed particle data
Next:numberCount = 100             ! print message every n events
Next:numberShowInfo = 1            ! print event information n times
Next:numberShowProcess = 1         ! print process record n times
Next:numberShowEvent = 0           ! print event record n times

Beams:idA = 11                     ! first beam, e+ = 11
Beams:idB = -11                    ! second beam, e- = -11
Beams:eCM = 380                    ! CM energy of collision

PartonLevel:ISR = on
PartonLevel:FSR = on

! 4) Hard process : photon collisions at 380
PhotonCollision:gmgm2qqbar = on
PhotonCollision:gmgm2ccbar = on
PhotonCollision:gmgm2bbbar = on
