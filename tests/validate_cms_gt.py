import pickle
import numpy as np
import argparse
import sys
import os

def validate(file_path):
    print(f"Opening {file_path}...")
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return False

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    num_events = len(data)
    print(f"Validating {num_events} events.")
    
    total_errors = 0

    for iev in range(num_events):
        errors = []
        event = data[iev]
        
        ytarget = event["ytarget"]
        # particle_feature_order:
        # 0: pid, 1: charge, 2: pt, 3: eta, 4: sin_phi, 5: cos_phi, 6: energy, ..., 13: particle_number
        
        pns = ytarget["particle_number"].astype(int)
        pids = ytarget["pid"].astype(int)
        energies = ytarget["energy"]
        
        unique_pns = np.unique(pns)
        
        # 1. Noise Check: Noise elements (ID 0) must have PID 0 and Energy 0
        if 0 in unique_pns:
            mask_noise = (pns == 0)
            noise_pids = pids[mask_noise]
            noise_energies = energies[mask_noise]
            
            # In CMS, elements with PN 0 might have non-zero PID/Energy if they are not associated with a CaloParticle
            # but are still part of the target? 
            # Actually, in postprocessing2.py, if len(caloparticles) == 0, ytarget stays 0.0 (filled with 0.0 initially).
            # So they should indeed be 0.
            if not np.all(noise_pids == 0):
                errors.append(f"Noise elements (PN 0) have non-zero PIDs: {np.unique(noise_pids[noise_pids != 0])}")
            if not np.all(noise_energies == 0):
                # Using a small threshold for float comparison if necessary, but here they are filled with 0.0
                if np.any(noise_energies > 1e-6):
                    errors.append(f"Noise elements (PN 0) have non-zero Energies!")

        # 2. Consistency Check for each particle
        for pn in unique_pns:
            if pn == 0:
                continue
            
            mask = (pns == pn)
            group_pids = pids[mask]
            group_energies = energies[mask]
            
            # 2.1 Representative Check: Every PN > 0 should have at least one element with PID > 0
            # In CMS, we merge CaloParticles and assign the merged properties to all elements associated ONLY to those CaloParticles.
            # Wait, the logic in postprocessing2.py is:
            # for each element, find associated caloparticles.
            # If > 0, merge them and set ytarget[ielem] = merged_cp.
            # So all elements with the same PN MUST have the same PID and Energy (since they come from the same merged CP).
            
            unique_group_pids = np.unique(group_pids)
            if len(unique_group_pids) > 1:
                errors.append(f"PN {pn} has inconsistent PIDs: {unique_group_pids}")
            
            if np.all(group_pids == 0):
                errors.append(f"PN {pn} has all PID=0.")

        if errors:
            print(f"\nEvent {iev} failed validation:")
            for err in errors[:10]:
                print(f"  - {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors)-10} more errors.")
            total_errors += len(errors)
        else:
            if iev % 100 == 0:
                print(f"Event {iev} passed.")

    if total_errors == 0:
        print("\nSUCCESS: All events passed ground truth validation.")
        return True
    else:
        print(f"\nFAILURE: Found {total_errors} consistency errors across {num_events} events.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    
    success = validate(args.input)
    sys.exit(0 if success else 1)
