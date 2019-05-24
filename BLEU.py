

# two references for one document
from nltk.translate.bleu_score import corpus_bleu
references = "test/IWSLT17.TED.tst2010.en-ro.ro.tok.txt"
 
candidates = "test/IWSLT17.TED.tst2010.en-ro.ENG-ROM.txt"

fileInput1 = open(references, 'r')

fileInput2 = open(candidates, 'r')


score = corpus_bleu(fileInput1, fileInput2)
print(score)
 
