import sys
import pickle
from DataFormats.FWLite import Events, Handle

class Expression:
    def __init__(self, label, edmtype, eval_list):
        self.label = label
        self.edmtype = edmtype
        self.eval_list = eval_list

        self.handle = Handle(self.edmtype)

    def get(self, event):
        event.getByLabel(self.label, self.handle)
        obj = self.handle.product()
        results = {}
        for eval_name, eval_item in self.eval_list:
            ret = eval(eval_item)
            results[eval_name] = ret
        return results

if __name__ == "__main__":
    filename = sys.argv[1]
    events = Events(filename)

    expressions = []
    expressions.append(Expression(
        "ak4PFJetsCHS",
        "vector<reco::PFJet>",
        [
            ("pt", "[o.pt() for o in obj]"),
            ("eta", "[o.eta() for o in obj]"),
            ("phi", "[o.phi() for o in obj]"),
            ("energy", "[o.energy() for o in obj]"),
        ]
    ))
    expressions.append(Expression(
        "ak4PFJetsPuppi",
        "vector<reco::PFJet>",
        [
            ("pt", "[o.pt() for o in obj]"),
            ("eta", "[o.eta() for o in obj]"),
            ("phi", "[o.phi() for o in obj]"),
            ("energy", "[o.energy() for o in obj]"),
        ]
    ))
    expressions.append(Expression(
        "pfMet",
        "vector<reco::PFMET>",
        [
            ("pt", "[o.pt() for o in obj]"),
            ("phi", "[o.phi() for o in obj]"),
        ]
    ))
    expressions.append(Expression(
        "pfMetPuppi",
        "vector<reco::PFMET>",
        [
            ("pt", "[o.pt() for o in obj]"),
            ("phi", "[o.phi() for o in obj]"),
        ]
    ))
    expressions.append(Expression(
        "particleFlow",
        "vector<reco::PFCandidate>",
        [
            ("pt", "[o.pt() for o in obj]"),
            ("eta", "[o.eta() for o in obj]"),
            ("phi", "[o.phi() for o in obj]"),
            ("energy", "[o.energy() for o in obj]"),
            ("pdgId", "[o.pdgId() for o in obj]"),
        ]
    ))
    
    evids = []
    for iev, event in enumerate(events):
        eid = event.object().id()
        eventId = (eid.run(), eid.luminosityBlock(), int(eid.event()))
        evids.append((eventId, iev))
    evids = sorted(evids, key=lambda x: x[0])

    #loop over events in a well-defined order
    all_results = []
    for _, iev in evids:
        event.to(iev)

        eid = event.object().id()
        eventId = (eid.run(), eid.luminosityBlock(), int(eid.event()))

        results = {}
        for expr in expressions:
            results[expr.label] = expr.get(event)
        all_results.append(results)
    pickle.dump(all_results, open("out.pkl", "wb"))

