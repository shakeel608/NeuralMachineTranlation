from bs4 import BeautifulSoup as bs
from nltk import word_tokenize 
import nltk
 

inputpath = "test/IWSLT17.TED.tst2010.en-ro.en.xml"
with open(inputpath, "r") as f:
    soup = bs(f.read(), "lxml")
    lines = [
      (int(j["id"]), j.text.strip())  for j in soup.findAll("seg")]
    # Ensure the same order
    #lines = sorted(lines, key=lambda x: x[0])
  # for x in soup.findAll("doc") 
#for i in lines:
	#print(i)
print(len(lines)) 	


outputpath = "test/IWSLT17.TED.tst2010.en-ro.en.txt"
with open(outputpath, "w") as f:
	for line in lines:
		f.write(str(line[1]))
		f.write("\n")
	
    
