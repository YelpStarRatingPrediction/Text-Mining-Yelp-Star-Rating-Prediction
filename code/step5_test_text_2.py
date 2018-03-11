import time
import re
from pandas import Series, DataFrame
import pandas as pd
import copy
import numpy as np
import nltk
import enchant
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.metrics import edit_distance

start = time.time()

class SpellingReplacer(object):
	def __init__(self, dict_name='en', max_dist=2):
		self.spell_dict = enchant.Dict(dict_name)
		self.max_dist = max_dist
	def replace(self, word):
		if self.spell_dict.check(word):
			return word
		suggestions = self.spell_dict.suggest(word)
		if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
			return suggestions[0]
		else:
			return word
replacer = SpellingReplacer()


def clean_text(x):
	# the underscore "_" actually stands for space " ". Hence replace "_" by " ".
	x=re.sub(r"_",' ',x)
	# delate all words containing numbers which are annoying
	x=re.sub(r" ([^\s]*[0-9]+[^\s]*)+ *",' ',x)
	x=re.sub(r"^[^\s]*[0-9]+[^\s]* *",' ',x)
	# replace the words containing 'not'
	x=re.sub(r"don't\s*",'not ',x)
	x=re.sub(r"doesn't\s*",'not ',x)
	x=re.sub(r"isn't\s*",'not ',x)
	x=re.sub(r"wasn't\s*",'not ',x)
	x=re.sub(r"not\s*",'not_',x)
	x=re.sub(r"not_so\s*",'not_so_',x)
	x=re.sub(r"\n",'',x)
	x=re.sub(",",' ',x)
	x=re.sub("'",' ',x)
	x=re.sub('"',' ',x)
	x=re.sub(r" +",' ',x)
	words = pd.Series(x.strip().split(' '))
	words = words[words!='']
	if len(words)==0:
		return('')
	# Lemmatizing words
	words = words.apply(lemmatizer.lemmatize)
	# Spelling correction
	words = words.apply(replacer.replace)
	x = " ".join(words)
	return(x.lower())

w = open("./data/test_text2.csv",'w')
f = open('./data/raw_text_test.csv','r')
w.write("Id,text\n")
f.readline()

for i in range(200000):
	f.readline()

a=0
f.readline()
for i in f:
	a += 1
	if a > 200000:
		break;
	text = i.split(',')
	if not str.isdigit(text[0]):
		print(str(a)+'  ,  '+text[0])
		print("Wrong!")
		break;
	else:
		w.write(str(text[0])+','+clean_text(text[1])+'\n')
	print(str(a)+'  ,  '+text[0])
w.close()
f.close()

end =time.time()
print("read: %f s" % (end - start))
