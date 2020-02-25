import ROOT

tf = ROOT.TFile("pfntuple.root")
tt = tf.Get("ana/pftree")

for ev in tt:
    caloparticle_pts = ev.caloparticle_pt

    caloparticle_to_cand = {}
    trackingparticle_to_cand = {}
    ncand_unmatched = 0
    for icand in range(len(ev.pfcandidate_pt)):
        idx_tp = ev.pfcandidate_idx_trackingparticle[icand] 
        idx_cp = ev.pfcandidate_idx_caloparticle[icand]
        if idx_tp != -1:
            trackingparticle_to_cand[idx_tp] = icand
        if idx_cp != -1:
            caloparticle_to_cand[idx_cp] = icand
        if idx_tp == -1 and idx_cp == -1:
            print("pfcand {} pid={} eta={:.2f}".format(icand, ev.pfcandidate_pdgid[icand], ev.pfcandidate_eta[icand]))
            ncand_unmatched += 1

    print("ncand={} ncand_unmatched={}".format(len(ev.pfcandidate_pt), ncand_unmatched))


    cands_seen = []
    cluster_to_caloparticle = {}
    for icalo in range(len(caloparticle_pts)):
        idx_cluster = ev.caloparticle_idx_cluster[icalo]
        if idx_cluster != -1:
            idx_cand = caloparticle_to_cand.get(icalo, -1)
            if idx_cand != -1:
                if idx_cand in cands_seen:
                    print("seen calo", idx_cand)
                else:
                    cands_seen.append(idx_cand)
            if not (idx_cluster) in cluster_to_caloparticle:
                cluster_to_caloparticle[idx_cluster] = []
            cluster_to_caloparticle[idx_cluster].append(icalo)

            print("icp={}/{}/{} pid={}/{} pt={:.2f}/{:.2f}/{:.2f} eta={:.2f}/{:.2f}/{:.2f} phi={:.2f}/{:.2f}/{:.2f}".format(
                icalo, idx_cluster, idx_cand,
                ev.caloparticle_pid[icalo], ev.pfcandidate_pdgid[idx_cand] if idx_cand!=-1 else 0,
                caloparticle_pts[icalo], ev.cluster_energy[idx_cluster], ev.pfcandidate_pt[idx_cand] if idx_cand!=-1 else 0,
                ev.caloparticle_eta[icalo], ev.cluster_eta[idx_cluster], ev.pfcandidate_eta[idx_cand] if idx_cand!=-1 else 0,
                ev.caloparticle_phi[icalo], ev.cluster_phi[idx_cluster], ev.pfcandidate_phi[idx_cand] if idx_cand!=-1 else 0,
            ))

    for cl, calos in cluster_to_caloparticle.items():
        if len(calos) > 1:
            print("cluster2calo", cl, len(calos))

    trackingparticle_pts = ev.trackingparticle_pt
    for itp in range(len(trackingparticle_pts)):
        idx_track = ev.trackingparticle_idx_track[itp]
        if idx_track != -1:
            idx_cand = trackingparticle_to_cand.get(itp, -1)
            if idx_cand in cands_seen:
                print("seen", idx_cand)
            print("itp={}/{}/{} pid={}/{} pt={:.2f}/{:.2f}/{:.2f} eta={:.2f}/{:.2f}/{:.2f} phi={:.2f}/{:.2f}/{:.2f}".format(
                itp, idx_track, idx_cand,
                ev.trackingparticle_pid[itp], ev.pfcandidate_pdgid[idx_cand] if idx_cand!=-1 else 0,
                trackingparticle_pts[itp], ev.track_pt[idx_track], ev.pfcandidate_pt[idx_cand] if idx_cand!=-1 else 0,
                ev.trackingparticle_eta[itp], ev.track_eta[idx_track], ev.pfcandidate_eta[idx_cand] if idx_cand!=-1 else 0,
                ev.trackingparticle_phi[itp], ev.track_phi[idx_track], ev.pfcandidate_phi[idx_cand] if idx_cand!=-1 else 0,
            ))
