import time

start = time.time()
w = open('./data/test_text.csv','w')
w.write("Id,text\n")
f1 = open("./data/test_text1.csv",'r')
f1.readline()
for i in f1:
	w.write(i)
f1.close()

f2 = open("./data/test_text2.csv",'r')
f2.readline()
for i in f2:
	w.write(i)
f2.close()

f3 = open("./data/test_text3.csv",'r')
f3.readline()
for i in f3:
	w.write(i)
f3.close()

f4 = open("./data/test_text4.csv",'r')
f4.readline()
for i in f4:
	w.write(i)
f4.close()

f5 = open("./data/test_text5.csv",'r')
f5.readline()
for i in f5:
	w.write(i)
f5.close()

w.close()


end =time.time()
print("read: %f s" % (end - start))
