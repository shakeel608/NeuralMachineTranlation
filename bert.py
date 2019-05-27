
import torch
import ast
import pytorch_pretrained_bert as bert
from pytorch_pretrained_bert import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
from itertools import repeat 


 

#embeddingsModel = "bert-base-multilingual-cased"
# see list of templates here https://github.com/huggingface/pytorch-pretrained-BERT#overview
embeddingsModel = "bert-base-uncased"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 32

# I make a tokenizer corresponding to my embeddings template
bert_tokenizer = BertTokenizer.from_pretrained(embeddingsModel, do_lower_case=False)

# I import the embeedings from the cache (or the net if we do not have them already)

bert_model = BertModel.from_pretrained(embeddingsModel)
bert_model = bert_model.to(device)

#bert_model.eval()

#lines = process("In the name of ALlah, the most beneficient and most merciful. 
#He is the Lord of the universe. To him we belog to him we shall return. He sent Mr. 
#Prophet Mohammad S.a.w as a messenger.")   
#print(lines) 

#Path To Files
inputpath_src = "BTEC-en-rom/train/train.tags.en-ro.en.clean.txt"
outputpath_src = "BTEC-en-rom/train/train.tags.en-ro.en.cls-sep.txt"

# Bert Tokenization, Sentences Segmentation first
# Put a [CLS] tag at the beginning of the text and a [SEP] tag at the end of the  each sentence
def process(data):
    line = ["[CLS]"]
    sentences = sent_tokenize(data)
    for i in range(len(sentences)):
        tokens = bert_tokenizer.tokenize(sentences[i])
        #line.append(tokens)
        #line.append("[SEP]")
        line += tokens + ["[SEP]"]
        #print(sentences[i])
    return line
   

#Load Source Language Files
fileInput_src = open(inputpath_src, 'r')
fileOutPut_src = open(outputpath_src, 'w')


 #Storing bert token of full corpus
bert_tokens_src = []

#Process Line by Line
for row in fileInput_src.readlines():
    line = process(row)           
    bert_tokens_src += line
    
    #print(line)
    #print("==============")
    #print >> fileOutPut_src, line
    fileOutPut_src.write(str(line)+"\n")
    

print("===============================================================================================================")


#Closing Files
fileInput_src.close()
fileOutPut_src.close()


#Reading String File into Tokens List again in order to prepare it for batch size
f = open(outputpath_src).read().splitlines()
bert_src_tokens = []

#Function Converting Tokens linewise to IDs
def convert_to_ids(line):
    #global X_train_tokens
    ids = bert_tokenizer.convert_tokens_to_ids(line)
    return ids

#For PAdding Sequences  =================================================Padding====================================================================
# long_lenth = 0
# padded_src_ids = src_ids.copy()

# #Find Lonest sequence length
# for i in src_ids:
#     if len(i) >= long_lenth:
#         long_lenth = len(i)
        
        
# #Padding Sequences with 0        
# def paddin_0():
#     for i,ids in enumerate(padded_src_ids):
#         padded_src_ids[i].extend(repeat(0, long_lenth - len(ids) ))
#         print(padded_src_ids[i])    

# #Function Call
# paddin_0()

print(bert_src_tokens.count('[SEP]'))   #From File 
print(bert_tokens_src.count('[SEP]'))


#Function to Genrate Bert Embeddings
def bert_embeddings(ids):
    #Transform tokens into identifiers (reminder: pytorch only handles integers)
    #ids_src=tokenizer.convert_tokens_to_ids(bert_src_tokens[:1024])
    
    #Transform ids into a pytorch tensor (representation in length is 64 bits of precision, 
    with torch.no_grad():
        ids = torch.tensor(ids, dtype=torch.long ,device = device) #Creates 1 D  Tensor of tokens


        #import os
        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"]="1"

        
        """unsqueeze is a function that adds a dimension of size 1 in the tensor.
        Some pytorch functions (including embeddings below) expect to have input 3D
        tensors while at the moment we have 2D tensors.
        So, we use unsqueeze to add a dimension of size 1 to the tensor."""       

		#Source Language

        ids=ids.unsqueeze(0)
    
    """calculation of all the embeddings of the source language""" 
    with torch.no_grad():
        embeddings_src = bert_model(ids)
        print(embeddings_src)
    
    return embeddings_src


  
  src_ids = []
#Create a Batch size of 520 sentences Lines at at Time 
for each_line in f:
    line = ast.literal_eval(each_line)
    line = [i.strip() for i in line]
    bert_src_tokens += line

    """Tokens to IDs"""
    ids = convert_to_ids(line)

    """Generate Bert Embeddings """
    source_embeddings = bert_embeddings(line)
    src_ids.append(ids)
    print("=============================================Lines======================================================")
    print(line)  
    print("===============================================IDs====================================================")
    print(ids)
    print("==============================================Len(IDs)=====================================================")
    print(len(ids))
    print("=============================================== Bert Embeddings===================================================")
    print(source_embeddings)
    print("=============================================== Lines===================================================")



# print(ls)    
# #X_train_tokens = [tokenizer.convert_tokens_to_ids(sent) for sent in f]
# results = torch.zeros((len(X_test_tokens), bert_model.config.hidden_size)).long()
# with torch.no_grad():
#     for stidx in range(0, len(X_test_tokens), batch_size):
#         X = X_test_tokens[stidx:stidx + batch_size]
#         X = torch.LongTensor(X)
#         _, pooled_output = bert_model(X)
#         results[stidx:stidx + batch_size,:] = pooled_output.cpu()

# Feed Batch Size of 50 sentences at a time to BERT MODEL for  creating emneddings 

# for line in padded_src_ids:    
#             """Generate Bert Embeddings using Batch Size of 50"""
#             source_embeddings = bert_embeddings(line)

# # line = [101, 10725, 10105]
# # source_embeddings = bert_embeddings(line)
# batch_size  = 32  
# with torch.no_grad():
#     for i in range(0, len(X_test_tokens), batch_size):
#         X = padded_src_ids[i:i + batch_size]
#         X = torch.LongTensor(X)
#         _, pooled_output = bert_model(X)
#         results[stidx:stidx + batch_size,:] = pooled_output.cpu()




print(source_embeddings)


#*************************************************CPU Version****************************************************#

#

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# #Function to Genrate Bert Embeddings
# def b_embeddings(batch):
#     #Transform tokens into identifiers (reminder: pytorch only handles integers)
#     #ids_src=tokenizer.convert_tokens_to_ids(bert_src_tokens[:1)
    
#     #Transform ids into a pytorch tensor (representation in length is 64 bits of precision, 
#     with torch.no_grad():
#         batch = torch.tensor(batch, dtype=torch.long, device=device) #Creates 1 D  Tensor of tokens
        
#         """unsqueeze is a function that adds a dimension of size 1 in the tensor.
#         Some pytorch functions (including embeddings below) expect to have input 3D
#         tensors while at the moment we have 2D tensors.
#         So, we use unsqueeze to add a dimension of size 1 to the tensor."""

# #Source Language
#         batch=batch.unsqueeze(0)
    
#     """calculation of all the embeddings of the source language
#     embeddings_src = embeddings(ids_src, output_all_encoded_layers=False).to('cuda')"""

# #import os
# #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# # os.environ["CUDA_VISIBLE_DEVICES"]="1" 
#     with torch.no_grad():
#         embeddings_src = embeddings(batch)
    
#     return embeddings_src


# for line in padded_src_ids:
#     ber_vector = b_embeddings(line)

    
# # Feed Batch Size of 50 sentences at a time to BERT MODEL for  creating emneddings      
# # for item in bert_tokens_src:
# #     batch += item    
# #     if item == '[SEP]':
# #         sentences +=1
# #         if sentences == 50:
# #             sentences = 0
# #             """Generate Bert Embeddings using Batch Size of 50"""
# #             source_embeddings = bert_embeddings(batch)
# #             break
       
    
    

 


# In[173]:
