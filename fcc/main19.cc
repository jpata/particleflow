// main19.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Modified by Joosep Pata to keep only PU
//g++ main19.cc -o main -I/home/joosep/pythia8308/include -I/home/joosep/HepMC3/hepmc3-install/include/ -L/home/joosep/HepMC3/hepmc3-install/lib/ -O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread  -L/home/joosep/pythia8308/lib -Wl,-rpath,/home/joosep/pythia8308/lib -lpythia8 -ldl -lHepMC3

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include <string>
using namespace Pythia8;

//==========================================================================

// Method to pick a number according to a Poissonian distribution.

int poisson(double nAvg, Rndm& rndm) {

  // Set maximum to avoid overflow.
  const int NMAX = 100;

  // Random number.
  double rPoisson = rndm.flat() * exp(nAvg);

  // Initialize.
  double rSum  = 0.;
  double rTerm = 1.;

  // Add to sum and check whether done.
  for (int i = 0; i < NMAX; ) {
    rSum += rTerm;
    if (rSum > rPoisson) return i;

    // Evaluate next term.
    ++i;
    rTerm *= nAvg / i;
  }

  // Emergency return.
  return NMAX;
}

//==========================================================================

int main(int argc, char *argv[]) {

  // Number of signal events to generate.
  int nEvent = 100;

  if (argc != 2) {
    std::cerr << "./main SEED" << std::endl;
    return 1;
  }
  
  std::string seedStr = std::string("Random:seed = ").append(std::string(argv[1]));

  // Average number of pileup events per signal event.
  double nPileupAvg = 10.0;
  
  // Shift each PU event by this time delta in time to mimic ee overlay
  double timeDelta = 0.5;
  
  Pythia8ToHepMC ToHepMC;
  ToHepMC.setNewFile("pythia.hepmc");

  // Signal generator instance.
  Pythia pythiaSignal;
  pythiaSignal.readFile("card.cmd");
  pythiaSignal.readString(seedStr.c_str());
  pythiaSignal.init();

  // Background generator instances copies settings and particle data.
  Pythia pythiaPileup;
  pythiaPileup.readFile("p8_ee_gg_ecm380.cmd");
  pythiaPileup.readString(seedStr.c_str());
  pythiaPileup.init();

  // One object where all individual events are to be collected.
  Event sumEvent;

  // Loop over events.
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

    HepMC3::GenEvent geneve;

    // Generate a signal event. Copy this event into sumEvent.
    if (!pythiaSignal.next()) continue;
    sumEvent = pythiaSignal.event;
    bool fill_result = ToHepMC.fill_next_event(pythiaSignal, &geneve);
    if (!fill_result) {
      std::cerr << "Error converting to HepMC" << std::endl;
      return 1;
    }

    // Select the number of pileup events to generate.
    int nPileup = poisson(nPileupAvg, pythiaPileup.rndm);

    // Generate a number of pileup events. Add them to sumEvent.
    for (int iPileup = 0; iPileup < nPileup; ++iPileup) {
      pythiaPileup.next();
      fill_result = ToHepMC.fill_next_event(pythiaPileup, &geneve);
      if (!fill_result) {
        std::cerr << "Error converting to HepMC" << std::endl;
        return 1;
      }
      for (int iPtcl=0; iPtcl < pythiaPileup.event.size(); iPtcl++) {
        auto& ptcl = pythiaPileup.event[iPtcl];
        double timeOffset = iPileup * timeDelta;
        ptcl.vProd(ptcl.xProd(), ptcl.yProd(), ptcl.zProd(), ptcl.tProd()+timeOffset);
      }
      sumEvent += pythiaPileup.event;
    }

    std::cout << "hepmc=" << geneve.particles().size() << " pythia=" << sumEvent.size() << std::endl;
    ToHepMC.output().write_event(geneve);
    
    // List first few events.
    if (iEvent < 5) {
      std::cout << "sumEvent" << std::endl;
      sumEvent.list();
    }


  // End of event loop
  }

  // Statistics. Histograms.
  pythiaSignal.stat();
  pythiaPileup.stat();

  return 0;
}
