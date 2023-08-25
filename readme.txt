The format of the train and test data json file is as follows:

- the top level object is an array of abstract objects

- abtract objects represent annotated clinical trial abstracts and have the following fields:
  -- abstract_id: id of clinical trial abstract
  -- sentences: array of sentence objects
  
- sentence objects represent sentences of an abstract and have the following fields:
  -- sentence_id: id of sentence (only uniuqe within abstract)
  -- words: array of strings
  -- entities: array of entity objects; entities of the sentence
  
- entity objects represent named entities and have the following fields:
  -- start_pos/end_pos: word offsets within sentence of start/end position of entity (both inclusive)
  -- label: label(class) of entity; 
  -- words: array of strings representing named entity
