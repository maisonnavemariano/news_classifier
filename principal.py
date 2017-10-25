import time
start = time.time()

textos_p       = 'pickles/textos.p'
target_p       = 'pickles/target.p'
target_names_p = 'pickles/target_names.p'
INPUT = '../../../dataset/noticias2013_reducido'
import numpy as np
import pickle
import re
def get_feature(feature):
	return [linea[len(feature):] for linea in lineas if linea.startswith(feature)]


try:
	target = pickle.load(open(target_p,'rb'))
	target_names = pickle.load(open(target_names_p, 'rb'))
except:
	lineas   = open(INPUT, 'r', encoding='utf-8').read().splitlines()
	sections = get_feature('sectionName: ')
	target = [1 if section=='Business' else 0 for section in sections]
	target_names = ['no_relevant','relevant']
	pickle.dump(target,open(target_p,'wb'))
	pickle.dump(target_names,open(target_names_p,'wb'))
try:
	textos = pickle.load(open(textos_p,'rb'))
except:
	lineas   = open(INPUT, 'r', encoding='utf-8').read().splitlines()

	titulos  = get_feature('webTitle: ')
	textos   = get_feature('bodytext: ')
	id       = get_feature('instanceNro: ')
	date     = get_feature('webPublicationDate: ')

	noticias = list(zip(id,date,titulos,textos))
	print('cantidad de noticias: {}'.format(len(noticias)))
	from nltk import word_tokenize
	from nltk.corpus import stopwords
	import string
	from bs4 import BeautifulSoup
	stopword_list = stopwords.words('english')
	stopword_file = '../../../dataset/stopwords.txt'
	stopword_list += open(stopword_file,'r',encoding='utf-8').read().splitlines()
	stopword_list = set(stopword_list)
	letters = set(string.ascii_letters)
	print(str(stopword_list))
	print('longitud: {}'.format(len(stopword_list)))
	def my_word_tokenize(text):
		texto = re.sub('&apos;',"'",texto)
		texto = re.sub('&quot;','"',texto)
		text = BeautifulSoup(text,'html.parser').get_text()
		lista_palabras = word_tokenize(text)
		lista_palabras = [palabra.lower() for palabra in lista_palabras]
		lista_palabras = [palabra for palabra in lista_palabras if palabra not in stopword_list]
		lista_palabras = [palabra for palabra in lista_palabras if palabra[0] in letters]
		return lista_palabras
	textos = [my_word_tokenize(texto)+my_word_tokenize(titulo) for (id,fecha,titulo,texto) in noticias]
	pickle.dump(textos,open(textos_p,'wb'))
from tfidf import corpus

texto1 = ['rio','danubio']
texto2 = ['rio', 'nilo']
texto3 = ['mar', 'mediterraneo']
texto4 = ['mar','caribe']
texto5 = ['piramides', 'egipto']

c = corpus(list(enumerate(textos)))

print('cantidad de noticias: {}'.format(len(c.textos_dict)))
#m = np.zeros(shape=(len(textos),len(textos)))
#for f in range(0,len(m)):
#	for row in range(0,len(m[0])):
#		m[f][row] =  c.similitud_docs_index(f,row)

#for fila in m:
#	cad = ""
#	for elem in fila:
#		cad+=str(elem)+","
#	print(cad[:-1])

end = time.time()
print(end - start)
#print(str(m))