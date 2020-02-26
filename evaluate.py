testFile="basicclassifier.py"
trainDir = "training/"
testDir = "testing/"
from sklearn import metrics
y_true=[]
y_pred=[]

exec(compile(open(testFile,"rb").read(),testFile,'exec'))
bc = Bayes_Classifier(trainDir)

iFileList = []
for fFileObj in os.walk(testDir + "/"):
	iFileList = fFileObj[2]
	break
length=len(iFileList)
for rating in range(length):
    if iFileList[rating][7]=="1":
        y_true.append("negative")
    elif iFileList[rating][7]=="5":
        y_true.append("positive")
    else:
        y_true.append("neutral")  
    
print("%d test reviews.",len(iFileList))
results = {"negative":0, "neutral":0, "positive":0}
print( "\nFile Classifications:")
for filename in iFileList:
    fileText = bc.loadFile(testDir + filename)
    result = bc.classify(fileText)
    y_pred.append(result)    
    print(filename,result)
    results[result] += 1
print("\nResults Summary:")
for r in results:
	print("%s: %d" % (r, results[r]))
print("confusion matrix")
print(metrics.confusion_matrix(y_true, y_pred))
print("precision and recall")
# Print the precision and recall, among other metrics
print(metrics.classification_report(y_true, y_pred, digits=3))