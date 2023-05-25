f = open("dev_.txt", "r")
l = open("dev.txt", "a")
for line in f:
    new = line[:-3]+'\t'+line[-2:]
    l.write(new)