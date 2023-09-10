import spacy
#print(dir(spacy.parts_of_speech))

l = ['a', 'b', 'c']
if 'A' in l:
    print('jo')



#nlp = spacy.load("en_core_web_sm")
#deps = list(nlp.get_pipe("tagger").labels)

#from spacy.glossary import GLOSSARY
#lookup_dict = GLOSSARY

#a = list(spacy.parts_of_speech.IDS.keys())
#for label in nlp.get_pipe("tagger").labels:
 #   print(label, " -- ", spacy.explain(label))

#nlp = spacy.load("en_core_web_sm")
#for label in nlp.get_pipe("parser").labels:
 #   print(label, " -- ", spacy.explain(label))