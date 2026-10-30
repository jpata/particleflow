# CLD ECAL and HCAL granularity scans

The 12-point CLD scan completed on 25 October 2026: six ECAL cell pitches
(5.1, 4, 3, 2.5, 2, 1.5 mm) in Z→ττ at 91 GeV and six HCAL pitches
(30, 25, 20, 15, 12, 10 mm) in ttbar at 365 GeV. Each point has 100
reconstructed ROOT files with 100 events, for 120,000 events in total.

The original Slurm controller `61753575` reached its 48-hour limit while
processing the final HCAL 10 mm point. Controller `61814971` finished the 19
remaining files and completed successfully. The geometry and production setup
are documented in `scripts/cld_ecal_readout/GRANULARITY_SCAN.md`.

- [Pandora photon, π⁰ and jet performance](performance/README.md)
- [Hit, track, cluster and PFO count validation](validation/README.md)
- [Paired event-display grids](events/README.md)

These results use the unchanged digitization and Pandora configuration. The
ECAL 5.1 mm scan point has the ECAL silicon 1 mm Geant step cap used at all
ECAL pitches, so it is not interchangeable with the older 5.1 mm production.
