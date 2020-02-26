import math, os, pickle, re
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import string

#PriorP_positive=0

class Bayes_Classifier:
    def __init__(self, trainDirectory = "movie_reviews/"):
       #if my pickled file exist then directly load that else go to train method and train data first 
        self.sfilename="database1.txt"
        if os.path.isfile(self.sfilename) and os.path.getsize(self.sfilename)>0:
            self.load(self.sfilename)
            
            
        else:
            f= open(self.sfilename,"w+")
            
            self.negativeDict = {}
            self.positiveDict = {}
            self.PriorP_positive=0
            self.PriorP_negative=0
            self.cn=0
            self.cp=0
            self.doc_class=""
            self.sText=''
            self.train_set()

    #start train get list of all the file names
    def train_set(self):
        trainDir = "training/"
        lFileList = []
        for fFileObj in os.walk("training/"):
            lFileList = fFileObj[2]
            break
        print(" Total training  reviews",len(lFileList))    
        #creating list of positive and negative reviews 
        positive_list=[]
        negative_list=[]
        length=len(lFileList)
        for rating in range(length):
            if lFileList[rating][7]=="1":
                negative_list.append(lFileList[rating])
                self.doc_class="negative"
                #loading the content of txt file with negative review and sending same to tokenize
                sTxt=self.loadFile(trainDir + lFileList[rating])
                self.tokensize(sTxt,self.doc_class)    
            elif lFileList[rating][7]=="5":
                positive_list.append(lFileList[rating])
                self.doc_class="positive"
                sTxt=self.loadFile(trainDir + lFileList[rating])
                self.tokensize(sTxt,self.doc_class)

        
        
        self.save()
        self.load(self.sfilename)
        self.prior_probability(negative_list,positive_list)
        
        
        print("total negative",self.cn)
        print("total positive",self.cp)

 #reading the files       
    def loadFile(self,filename):
        
        
            f = open(filename, 'rb')
            sTxt = f.read().decode('utf-8', errors='ignore')
        
            
            f.close()
            translator = str.maketrans('', '', string.punctuation)
            sTxt=sTxt.translate(translator)
            return sTxt
            
            
    def tokensize(self,sText,doc_class):
        #splitting the text in file
        stop_words = set(stopwords.words('english'))
        
        lTokens = []
        sToken = ""
        for c in sText:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
                sToken += c
                
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))
                           

        if sToken != "":
            lTokens.append(sToken)
        #removing stop words    
        lTokens = [w for w in lTokens if not w in stop_words]    
        for word in lTokens:
            self.build_dictionary(word,self.doc_class)
   #create dictionary add each word exacted above 

    def build_dictionary(self,word,doc_class):
         #if word exist update the count else update that word with count 1
        if self.doc_class=="negative":
                if self.negativeDict.get(word,0)==0:
                    self.negativeDict.update({word:1})
                    self.cn=self.cn+1
                elif self.negativeDict.get(word,0)>0:
                    self.negativeDict[word]=self.negativeDict[word]+1 
                    self.cn=self.cn+1

        if self.doc_class=="positive":
            if self.positiveDict.get(word,0)==0:
                self.positiveDict.update({word:1})
                self.cp=self.cp+1
                
            elif self.positiveDict.get(word,0)>0:
                self.positiveDict[word]=self.positiveDict[word]+1 
                self.cp=self.cp+1
      
        
    #pickling        
            
    def save(self):
        f = open(self.sfilename,"wb")
        pickle.dump(self,f)
        f.close() 
        
    def load(self, sFilename):
        f = open(sFilename, "rb")
        u = pickle.Unpickler(f)
        dObj = u.load()
        self.__dict__.update(dObj.__dict__)
        f.close()
        return dObj
#calculating prior probability

    
    def prior_probability(self,negative_list,positive_list):
        lengthN=len(negative_list)
        lengthP=len(positive_list)
        print("dic P length",lengthP)
        print("dic n length",lengthN)
        totalcount=lengthN+lengthP
        self.PriorP_positive=np.log(lengthP/totalcount)
        self.PriorP_negative=np.log(lengthN/totalcount)
        
        
        
    def classify(self,filetext):
        
        obj1=self.load(self.sfilename)
        
        class_review=["positive","negative"]
        result=[]
        lTokens = []
        sToken = ""
        #removing punctuations
        translator = str.maketrans('', '', string.punctuation)
        filetext.translate(translator)
        stop_words = set(stopwords.words('english'))
        #tokenizing the text in the traing file
        for c in filetext:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
                sToken += c
                
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))
        if sToken != "":
            lTokens.append(sToken)
            #removing stopwords
        lTokens = [w for w in lTokens if not w in stop_words]    
        lenp=len(obj1.positiveDict)
        
        lenn=len(obj1.negativeDict)
        for x in class_review:
            con_prob_positive=0 
            con_prob_negative=0
        #calculating conditional probability for each word
            if x=="positive":
                for word in lTokens:
                    count=obj1.positiveDict.get(word,0)
                    
                    Pword=np.log((count+1)/(self.cp+lenp))
                    con_prob_positive=con_prob_positive+Pword
                result.append(con_prob_positive+self.PriorP_positive)
                
            elif x=="negative":
                for word in lTokens:
                    count=obj1.negativeDict.get(word,0)
                    
                    Pword=np.log((count+1)/(self.cn+lenn))
                    con_prob_negative=con_prob_negative+Pword
                result.append(con_prob_negative+self.PriorP_negative)

        #checking and returning the result to main file evaluate.py
        if result[1]-result[0]<0.5 and result[1]-result[0]>-0.5:
            return "neutral"
        else:
            if(result[1]<result[0]):
                return "positive"
            if(result[1]>result[0]):
                return "negative"
        
   

                

#a=Bayes_Classifier()            
#a.train_set()
#a.classify("i live movie")
