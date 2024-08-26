#!/bin/bash

source /opt/hepsim.sh

rm -f MinBias.root MinBias.pileup

/opt/hepsim/delphes-local/DelphesPythia8 /opt/hepsim/delphes/cards/converter_card.tcl generatePileUpCMS.cmnd MinBias.root
root2pileup MinBias.pileup MinBias.root
rm -f MinBias.root
