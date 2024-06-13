import pickle
import sys
import bz2

try:
    data = pickle.load(bz2.BZ2File(sys.argv[1], "rb"), encoding="iso-8859-1")
except Exception:
    print(sys.argv[1])
