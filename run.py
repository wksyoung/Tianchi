import pickle
with open('dictionary.dat','rb') as f:
    dictionary = pickle.load(f)
print(dictionary)