import pickle
import sys
import tqdm

from DataFormats.FWLite import Events, Handle


class Expression:
    def __init__(self, label, edmtype, eval_list):
        self.label = label
        self.edmtype = edmtype
        self.eval_list = eval_list

        self.handle = Handle(self.edmtype)

    def get(self, event):
        event.getByLabel(self.label, self.handle)
        obj = self.handle.product()  # noqa F841
        results = {}
        for eval_name, eval_item in self.eval_list:
            ret = eval(eval_item)
            results[eval_name] = ret
        return results


if __name__ == "__main__":
    filename = sys.argv[1]
    outfilename = sys.argv[2]
    events = Events(filename)

    expressions = []
    expressions.append(
        Expression(
            "slimmedJets",
            "vector<pat::Jet>",
            [
                ("pt", "[o.pt() for o in obj]"),
                ("eta", "[o.eta() for o in obj]"),
                ("phi", "[o.phi() for o in obj]"),
                ("energy", "[o.energy() for o in obj]"),
            ],
        )
    )
    expressions.append(
        Expression(
            "slimmedJetsPuppi",
            "vector<pat::Jet>",
            [
                ("pt", "[o.pt() for o in obj]"),
                ("eta", "[o.eta() for o in obj]"),
                ("phi", "[o.phi() for o in obj]"),
                ("energy", "[o.energy() for o in obj]"),
            ],
        )
    )
    expressions.append(
        Expression(
            "slimmedMETs",
            "vector<pat::MET>",
            [
                ("pt", "[o.pt() for o in obj]"),
                ("phi", "[o.phi() for o in obj]"),
            ],
        )
    )
    expressions.append(
        Expression(
            "slimmedMETsPuppi",
            "vector<pat::MET>",
            [
                ("pt", "[o.pt() for o in obj]"),
                ("phi", "[o.phi() for o in obj]"),
            ],
        )
    )
    expressions.append(
        Expression(
            "slimmedGenJets",
            "vector<reco::GenJet>",
            [
                ("pt", "[o.pt() for o in obj]"),
                ("eta", "[o.eta() for o in obj]"),
                ("phi", "[o.phi() for o in obj]"),
                ("energy", "[o.energy() for o in obj]"),
            ],
        )
    )
    expressions.append(
        Expression(
            "genMetTrue",
            "vector<reco::GenMET>",
            [
                ("pt", "[o.pt() for o in obj]"),
                ("phi", "[o.phi() for o in obj]"),
            ],
        )
    )

    expressions.append(
        Expression(
            "packedPFCandidates",
            "vector<pat::PackedCandidate>",
            [
                ("pt", "[o.pt() for o in obj]"),
                ("eta", "[o.eta() for o in obj]"),
                ("phi", "[o.phi() for o in obj]"),
                ("energy", "[o.energy() for o in obj]"),
                ("pdgId", "[o.pdgId() for o in obj]"),
            ],
        )
    )

    expressions.append(
        Expression(
            "prunedGenParticles",
            "vector<reco::GenParticle>",
            [
                ("pt", "[o.pt() for o in obj]"),
                ("eta", "[o.eta() for o in obj]"),
                ("phi", "[o.phi() for o in obj]"),
                ("energy", "[o.energy() for o in obj]"),
                ("pdgId", "[o.pdgId() for o in obj]"),
                ("status", "[o.status() for o in obj]"),
            ],
        )
    )

    evids = []
    for iev, event in enumerate(events):
        eid = event.object().id()
        eventId = (eid.run(), eid.luminosityBlock(), int(eid.event()))
        evids.append((eventId, iev))
    evids = sorted(evids, key=lambda x: x[0])

    # loop over events in a well-defined order
    all_results = []
    for _, iev in tqdm.tqdm(evids):
        event.to(iev)

        eid = event.object().id()
        eventId = (eid.run(), eid.luminosityBlock(), int(eid.event()))

        results = {}
        for expr in expressions:
            results[expr.label] = expr.get(event)
        all_results.append(results)
    pickle.dump(all_results, open(outfilename, "wb"))
