a = open("export.csv", "r")
Lines = a.readlines()

for line in Lines:
    vec=line.split(",")
    if( len(vec) != 1441 ):
        print(len(vec))