import numpy as np
 
dataset = [['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']]
 
attributes=6
 
 
hypothesis = ['0'] * attributes
 
for i in range(0,len(dataset)):
    if dataset[i][attributes]=='Yes':
        for j in range(0,attributes):
            hypothesis[j] = dataset[i][j]
        break
 
 
print("\n Find S: Finding a Maximally Specific Hypothesis\n")
 
for i in range(0,len(dataset)):
    if dataset[i][attributes]=='Yes':
        for j in range(0,attributes):
            print(dataset[i][j], end=' ')
            if dataset[i][j]!=hypothesis[j]:
                hypothesis[j]='?'
            else :
                hypothesis[j]= dataset[i][j]
    print("\n\nFor Training instance No:{} the hypothesis is ".format(i), hypothesis)
    print("\n")
 
print("\n The Maximally Specific Hypothesis for a given Training Examples :\n")
print(hypothesis)
