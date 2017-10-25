import numpy as np
import math
import pickle

inverse_index_p  = 'pickles/inverse_index.p'
idf_p            = 'pickles/idf.p'
textos_dict_p    = 'pickles/textos_dict.p'

feature_names_p  = 'pickles/feature_names.p'
data_p           = 'pickles/data.p'

class corpus(object):
	def __init__(self, lista_textos_filtrados_con_id):
		from collections import defaultdict
		cuenta_palabra = defaultdict(int)
		textos_con_id = lista_textos_filtrados_con_id
		# Textos
		print('[OK] generamos diccionario de textos.')
		try:
			self.textos_dict = pickle.load(open(textos_dict_p,'rb'))
		except:
			print('[WARNING] pickle de textos no encontrado.')
			self.textos_dict = dict([(texto[0],texto) for texto in textos_con_id])
			pickle.dump(self.textos_dict,open(textos_dict_p,'wb'))
		# Indice Palabras
		print('[OK] generamos indice de terminos.')
		try:
			self.index_terminos = pickle.load(open(feature_names_p, 'rb'))
		except:
			print('[WARNING] pickle de indices no encontrado.')
			cjto_palabras = set()
			for texto in textos_con_id:
				for palabra in texto[1]:
					cuenta_palabra[palabra]+=1
					if cuenta_palabra[palabra]>=4:
						cjto_palabras.add(palabra)
			self.index_terminos = np.array(list(cjto_palabras))
			writer = open('palabras_frecuentes','w',encoding='utf-8')
			for termino in self.index_terminos:
				aparicion = len([1 for texto in textos_con_id if termino in texto[1]])
				if aparicion/len(textos_con_id)>0.8:
					writer.write('{}: {}'.format(termino,aparicion/len(textos_con_id)))
			writer.close()
			pickle.dump(self.index_terminos,open(feature_names_p,'wb'))
		# Indice inverso
		print('[OK] generamos indice inverso de terminos.')
		try:
			self.inverse_index = pickle.load(open(inverse_index_p,'rb'))
		except:
			print('[WARNING] pickle de indice inverso no encontrado.')
			self.inverse_index = dict([(palabra,indice) for (indice,palabra) in enumerate(self.index_terminos)])
			pickle.dump(self.inverse_index,open(inverse_index_p,'wb'))
		# idf
		print('[OK] generamos vector idf.')
		try:
			self.idf = pickle.load(open(idf_p,'rb'))
		except:
			print('[WARNING] pickle de idf no encontrado.')
			idf = []
			for term in self.index_terminos:
				idf.append(len([texto for texto in textos_con_id if term in texto[1]]))
			self.idf = np.array([math.log10(len(self.textos_dict)/valor) for valor in idf])
			pickle.dump(self.idf,open(idf_p,'wb'))
		print('[OK] generamos matriz de datos.')
		try:
			data = pickle.load(open('data_p','rb'))
		except:
			print('[WARNING] pickle de matriz de datos no encontrado.')
			data = zeros(shape=(len(textos_con_id), len(self.index_terminos)) )
			index = 0
			for (id,texto) in textos_con_id:
				data[index] = np.array(get_vect(texto))
				index += 1

	def __normalize(self,vec1):
		vec1_pow2 = [elem*elem for elem in vec1]
		module = math.sqrt(sum(vec1_pow2))
		return [elem/module for elem in vec1]

	def similitud_docs(self,text_1,text_2):
		tf_1 = np.array([text_1.count(term) for term in self.index_terminos])
		tf_2 = np.array([text_2.count(term) for term in self.index_terminos])
		tf_idf_1 = self.__normalize(np.array([tf*self.idf[indice] for (indice,tf) in enumerate(tf_1)]))
		tf_idf_2 = self.__normalize(np.array([tf*self.idf[indice] for (indice,tf) in enumerate(tf_2)]))
		return np.dot(tf_idf_1,tf_idf_2)
	def similitud_docs_index(self,index1, index2):
		tf_1 = np.array([self.textos_dict[index1][1].count(term) for term in self.index_terminos])
		tf_2 = np.array([self.textos_dict[index2][1].count(term) for term in self.index_terminos])
		tf_idf_1 = self.__normalize(np.array([tf*self.idf[indice] for (indice,tf) in enumerate(tf_1)]))
		tf_idf_2 = self.__normalize(np.array([tf*self.idf[indice] for (indice,tf) in enumerate(tf_2)]))
		return np.dot(tf_idf_1,tf_idf_2)
	def get_vect(self,text_1):
		tf_1 = np.array([text_1.count(term) for term in self.index_terminos])
		tf_idf_1 = self.__normalize(np.array([tf*self.idf[indice] for (indice,tf) in enumerate(tf_1)]))
		return tf_idf_1
	def get_vect_index(self,index_1):
		tf_1 = np.array([self.textos_dict[index1][1].count(term) for term in self.index_terminos])
		tf_idf_1 = self.__normalize(np.array([tf*self.idf[indice] for (indice,tf) in enumerate(tf_1)]))
		return tf_idf_1
