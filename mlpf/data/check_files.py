import os

lines = open("args.txt").readlines()
for line in lines:
    args = line.strip().split()
    if not os.path.isfile("{}/pfntuple_{}.root".format(args[0], args[1])):
        print(line.strip())
