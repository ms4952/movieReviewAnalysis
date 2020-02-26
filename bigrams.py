import math, os, pickle, re
import numpy as np
import nltk
from nltk.collocations import *
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


#PriorP_positive=0

class Bayes_Classifier:
    def __init__(self, trainDirectory = "movie_reviews/"):
        #if my pickled file exist then directly load that else go to train method and train data first
        self.sfilename="databasebest.txt"
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
        trainDir = "testing/"
        lFileList = []
        for fFileObj in os.walk("testing/"):
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

 #reading        
    def loadFile(self,filename):
        
            #f = open(filename, "r")
            f = open(filename, 'r', encoding='latin-1') 
            sTxt = f.read()
            f.close()
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
        lTokens = [w for w in lTokens if not w in stop_words]    
        self.bigram(lTokens)    
   #create pair for tokens extracted from text 
    def bigram(self,lTokens):
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(lTokens)
        all_pair=finder.nbest(bigram_measures.pmi,30)  
        for pair in all_pair:
            self.build_dictionary(pair,self.doc_class)
    #create dictionary add each pair exacted above
    def build_dictionary(self,pair,doc_class):
        if self.doc_class=="negative":
                if self.negativeDict.get(pair,0)==0:
                    self.negativeDict.update({pair:1})
                    self.cn=self.cn+1
                elif self.negativeDict.get(pair,0)>0:
                    self.negativeDict[pair]=self.negativeDict[pair]+1 
                    self.cn=self.cn+1

        if self.doc_class=="positive":
            if self.positiveDict.get(pair,0)==0:
                self.positiveDict.update({pair:1})
                self.cp=self.cp+1
            elif self.positiveDict.get(pair,0)>0:
                self.positiveDict[pair]=self.positiveDict[pair]+1 
                self.cp=self.cp+1
      
        
        
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

    
    def prior_probability(self,negative_list,positive_list):
        lengthN=len(negative_list)
        lengthP=len(positive_list)
        print("P length of dict",lengthP)
        print("n length of dict",lengthN)
        totalcount=lengthN+lengthP
        self.PriorP_positive=np.log(lengthP/totalcount)
        self.PriorP_negative=np.log(lengthN/totalcount)
        print("in pp function ","neg",self.PriorP_negative,"\npos",self.PriorP_positive)
        
        
        
        
   

    def classify(self,filetext):
        
        stop_words = set(stopwords.words('english'))
        obj1=self.load(self.sfilename)
        print("positive dict",obj1.positiveDict)
        print("\n \n negaitive dict",obj1.negativeDict)
        self.prior_probability(obj1.negativeDict,obj1.positiveDict)
        class_review=["positive","negative"]
        result=[]
        lTokens = []
        sToken = ""
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
        lTokens = [w for w in lTokens if not w in stop_words]    
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(lTokens)
        all_pair=finder.nbest(bigram_measures.pmi, 10)  
        
        lenp=len(obj1.positiveDict)
        lenn=len(obj1.negativeDict)
        
        
        for x in class_review:
            con_prob_positive=0 
            con_prob_negative=0
        
            if x=="positive":
                for pair in all_pair:
                    count=obj1.positiveDict.get(pair,0)
                    
                    Pword=np.log((count+1)/(self.cp+lenp))
                    con_prob_positive=con_prob_positive+Pword
                result.append(con_prob_positive+self.PriorP_positive) 
            elif x=="negative":
                for pair in all_pair:
                    count=obj1.negativeDict.get(pair,0)
                    
                    Pword=np.log((count+1)/(self.cn+lenn))
                    con_prob_negative=con_prob_negative+Pword
                result.append(con_prob_negative+self.PriorP_negative)
               

        if result[1]-result[0]<1 and result[1]-result[0]>-1:
            return "neutral"
        else:
            if(result[1]<result[0]):
                return "positive"
            if(result[1]>result[0]):
                return "negative"
        
   
            
   
                

