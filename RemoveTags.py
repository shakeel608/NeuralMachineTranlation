from bs4 import BeautifulSoup as bs
from nltk import word_tokenize 
import nltk
import re 
 
#Tags to remove
#<title>(.*?)</title> <talkid>1</talkid> <speaker>Al Gore</speaker> <keywords> <url> <reviewer></reviewer> <translator></translator>

TAG_1 = re.compile(r'<title>(.*?)</title>')
TAG_2 = re.compile(r'<description>(.*?)</description>')
TAG_3 = re.compile(r'<talkid>(\b\d+(?:\.\d+)?\s+)</<talkid>')
TAG_4 = re.compile(r'<speaker>(.*?)</speaker>')
TAG_5 = re.compile(r'<keywords>(.*?)</keywords>')
TAG_6 = re.compile(r'<url>(.*?)</url>')
TAG_7 = re.compile(r'<reviewer(.*?)>(.*?)</reviewer>')
TAG_8 = re.compile(r'<translator(.*?)>(.*?)</translator>')
removingdigits = re.compile(r'\d+')
clean = re.compile(r'<.*?>')



def remove_tags(text):
    text1 = TAG_1.sub('', text)
    text2 = TAG_2.sub('', text1)
    text3 = TAG_3.sub('', text2)
    text4 = TAG_4.sub('', text3)
    text5 = TAG_5.sub('', text4)
    text6 = TAG_6.sub('', text5)
    text7 = TAG_7.sub('', text6)
    text8 = TAG_8.sub('', text7)
    text9 = removingdigits.sub('', text8)
    #text10 = removingdigits.sub('', text9)
    #if re.match(r'^\s*$', text10):
    return clean.sub('', text9)



inputpath1 = "train/train.tags.en-ro.en"
outputpath1 = "train/train.tags.en-ro.en.txt"
fileInput1 = open(inputpath1, 'r')
fileOutPut1 = open(outputpath1, 'w')

for row in fileInput1.readlines():
	line = remove_tags(row)
	print(line)
	print("==============")
	fileOutPut1.write(line)
      	


fileInput1.close()
fileOutPut1.close()

inputpath2 = "train/train.tags.en-ro.ro"
outputpath2 = "train/train.tags.en-ro.ro.txt"
fileInput2 = open(inputpath2, 'r')
fileOutPut2 = open(outputpath2, 'w')

for row in fileInput2.readlines():
	line = remove_tags(row)
	print(line)
	fileOutPut2.write(line)

fileInput2.close()
fileOutPut2.close()


#Removing Empty lines 


with open(outputpath1) as filehandle:
        lines = filehandle.readlines()

with open(outputpath1, 'w') as filehandle:
       lines = filter(lambda x: x.strip(), lines)
       filehandle.writelines(lines)  

with open(outputpath2) as filehandle:
        lines = filehandle.readlines()

with open(outputpath2, 'w') as filehandle:
       lines = filter(lambda x: x.strip(), lines)
       filehandle.writelines(lines)  


#Removing Tags
#def remove_html_tags(text):
#    """Remove html tags from a string"""
 #   #clean = re.compile('<.*?>')
 #   return re.sub(clean, '', text)
    #return str(clean)













    
	



