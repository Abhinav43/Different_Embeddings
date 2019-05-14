#We see that the output dictionary has a word_emb, which seems to be what we need.
import tensorflow_hub as hub
import tensorflow as tf

word_to_embed = "dog"

elmo = hub.Module("https://tfhub.dev/google/elmo/2")
embedding_tensor = elmo([word_to_embed], as_dict=True)["word_emb"] # Use as_dict because I want the whole dict, then I select "word_emb"

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  embedding = sess.run(embedding_tensor)
  print(embedding.shape)
  
  
  
  #If I want to run multiple words, I can just pass in a list
  
  import tensorflow_hub as hub
import tensorflow as tf

words_to_embed = ["dog", "cat", "sloth"] # <-- it's a list now, and the name changed

elmo = hub.Module("https://tfhub.dev/google/elmo/2")
embedding_tensor = elmo(words_to_embed, as_dict=True)["word_emb"] # <-- passing in a list instead of [word]

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  embedding = sess.run(embedding_tensor)
  print(embedding.shape)
  
  
  #If I don’t specify as_dict, then it just displays default instead of word_emb.
  
  import tensorflow_hub as hub
import tensorflow as tf

words_to_embed = ["dog", "cat", "sloth"] 

elmo = hub.Module("https://tfhub.dev/google/elmo/2")
embedding_tensor = elmo(words_to_embed) # <-- removed other params

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  embedding = sess.run(embedding_tensor)
  print(embedding.shape)
