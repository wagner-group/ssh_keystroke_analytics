import random

def gen(N=10000, U=20):
    data = {i: [] for i in range(U)}
    for i in range(N):
        user = random.randint(0,U-1)
        timestamp = 1683162000 + random.randint(0,86399)
        sol = [(random.randint(16,127), random.randint(8,63)) for _ in range(random.randint(8,512))]
        sot = [random.randint(0,100000)/100000 for _ in sol]
        position = random.randint(0,512) / 1024
        data[user].append("{}\t{}\t0\t0\t{}\t{}\t{}\n".format(user, timestamp, ','.join(str(v) for v in sot), ','.join("{}:{}".format(v[0], v[0]+v[1]) for v in sol), position))

    data = sorted(list(data.items()), key=lambda x: -len(x[1]))
    for i in range(10):
        with open(str(data[i][0]), "w") as outfile:
            outfile.write("".join(data[i][1]))

    with open("others", "w") as outfile:
        for i in range(11,20):
            outfile.write("".join(data[i][1]))

    
    with open("authentication_manifest", "w") as outfile:
        for i in range(10):
            outfile.write("{}\t{}\n".format(data[i][0], len(data[i][1])))
        others = sum(len(data[i][1]) for i in range(11,20))
        outfile.write("others\t{}\n".format(others))


gen()
    
