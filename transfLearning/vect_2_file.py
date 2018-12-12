import os

trg = open("allvect.csv","w")
for file in os.listdir("image_vectors/"):
	src = open(os.path.join("image_vectors/",file),"r")
	line = src.readline()
	line = line[:-1]
	while line:
		trg.write(line+"\t")
		line = src.readline()
		line = line[:-1]
	trg.write("\n")
	src.close()
trg.close()
