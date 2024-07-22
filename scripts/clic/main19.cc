// main19.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Modified by Joosep Pata to keep only PU
//g++ main19.cc -o main -I/home/joosep/pythia8308/include -I/home/joosep/HepMC3/hepmc3-install/include/ -L/home/joosep/HepMC3/hepmc3-install/lib/ -O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread  -L/home/joosep/pythia8309/lib -Wl,-rpath,/home/joosep/pythia8309/lib -lpythia8 -ldl -lHepMC3

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include <string>
#include <cstdlib>

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

  if (argc != 3) {
    std::cerr << "./main SEED NPU" << std::endl;
    return 1;
  }

  std::string seedStr = std::string("Random:seed = ").append(std::string(argv[1]));

  // Average number of pileup events per signal event.
  double nPileupAvg = atoi(argv[2]);

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
  pythiaPileup.readFile("card_pu.cmd");
  pythiaPileup.readString(seedStr.c_str());
  pythiaPileup.init();

  // Loop over events.
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
    HepMC3::GenEvent geneve;

    // Select the number of pileup events to generate.
    int nPileup = poisson(nPileupAvg, pythiaPileup.rndm);

    // create a random index permutation from [0, nPileup)
    std::vector<int> puVectorInds;
    for (int npu=0; npu<nPileup; npu++) {
      puVectorInds.push_back(npu);
    }
    pythiaPileup.rndm.shuffle(puVectorInds);

    // Generate a number of pileup events. Add them to sumEvent.
    for (int iPileup = 0; iPileup < nPileup; ++iPileup) {

      //generate a signal event if the permutation value is 0, otherwise generate a pileup event
      auto& pythiaSigOrPU = (puVectorInds[iPileup] == 0) ? pythiaSignal : pythiaPileup;
      pythiaSigOrPU.next();

      for (int iPtcl=0; iPtcl < pythiaSigOrPU.event.size(); iPtcl++) {
        auto& ptcl = pythiaSigOrPU.event[iPtcl];
        double timeOffset = iPileup * timeDelta;
        ptcl.vProd(ptcl.xProd(), ptcl.yProd(), ptcl.zProd(), ptcl.tProd()+timeOffset);
      }

      bool fill_result = ToHepMC.fill_next_event(pythiaSigOrPU, &geneve);
      if (!fill_result) {
        std::cerr << "Error converting to HepMC" << std::endl;
        return 1;
      }
    }

    std::cout << "hepmc=" << geneve.particles().size() << std::endl;
    ToHepMC.output().write_event(geneve);

  }
  // End of event loop

  return 0;
}
