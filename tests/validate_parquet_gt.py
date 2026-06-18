"""
Spec: Validation script for Parquet-based ground truth files. Verifies: Continuous PN range, noise zeroing, presence of representative tracks/clusters for every PN, and cross-collection consistency between hits and high-level objects. Requires an external '.parquet' file (e.g., from 'scripts/fetch_test_data_cld.sh' or 'scripts/local_test_cld.sh').
"""
