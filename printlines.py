from bs4 import BeautifulSoup as bs
from nltk import word_tokenize 
import nltk
import re 
 

inputpath1 = "train/train.tags.en-ro.en"
outputpath1 = "test/IWSLT17.TED.tst2010.en-ro.en.clean.txt"
fileInput1 = open(inputpath1, 'r')
#fileOutPut1 = open(outputpath1, 'w')
total1 = 0
#for row in fileInput1.readlines():
	#total1 += 1
#	line = bs(row, 'lxml').text
	
 
	#print(line)
	#fileOutPut1.write(line)
      	




#fileInput1.close()
#fileOutPut1.close()

inputpath2 = "train/train.tags.en-ro.ro"
outputpath2 = "test/IWSLT17.TED.tst2010.en-ro.ro.clean.txt"
#fileInPut2 = open(inputpath2, 'r')
#fileOutPut2 = open(outputpath2, 'w')
total2 = 0
#for row in fileInPut2.readlines():
	#total2 += 1
	#print(row)
	#print("======")
	#line = bs(row, 'lxml').text
	#fileOutPut2.write(line)

	
#fileInPut2.close()
#fileOutPut2.close()

fileOutPut1 = open(outputpath1, 'r')
fileOutPut2 = open(outputpath2, 'r')
for row in fileOutPut1.readlines():
	total1 += 1
	#print(line)
for row in fileOutPut2.readlines():
	total2 += 1
	#print(line)

print(total1)
print(total2)
#Removing Spaces


















    
	



