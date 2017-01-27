# -*- coding: utf8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import xml.etree.cElementTree as ET
import logging
import os
import re
from pandas import DataFrame
import pandas as pd
import Stemmer
from scipy.sparse import csr_matrix

logging.basicConfig(format = u'%(filename)s %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.DEBUG)


stop_words = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его',
              'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
              'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже',
              'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом',
              'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их',
              'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда',
              'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой',
              'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два',
              'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая',
              'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть',
              'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между']
def parse(inputFile) :
    df = pd.DataFrame ( columns=('id', 'category', 'title'))

    file_path = inputFile
    context = ET.iterparse(file_path, events=("start", "end"))

    context = iter(context)
    on_page_tag = False
    count = 0
    row_id = ''
    row_category =''
    row_title =''

    for  event, elem in context:
        tag = elem.tag
        value = elem.text

        if event == 'start' :

            if tag == 'page' :
                on_page_tag = True

            elif tag == 'id' :
                if on_page_tag :
                    count += 1
                    if value is not None:
                        row_id = str(value)
                        if count % 5000 == 0 :
                            logging.info('completed ' + str(count))

            elif tag == 'title' :
                if on_page_tag :
                    if value is not None:
                        row_title = str(value)

            elif tag == 'category' :
                if on_page_tag :
                    if value is not None:
                        row_category = str(value)
                        if ((row_id != '') & (row_title != '')) :
                            rowt = dict(zip(['id', 'category', 'title'], [row_id, row_category, row_title]))
                            row_ser = pd.Series(rowt)
                            row_ser.name = row_id
                            df = df.append(row_ser)

        if event == 'end' :
            if tag == 'page' :
                on_page_tag = False
                row_id = row_category = row_title = ''

        elem.clear()

    return df

def clean_category(dataf_path) :
    df = pd.read_pickle(dataf_path)

    # dftimeMinusCat=df.groupby(['category']).filter(lambda x: x['category'].value_counts() <= 25)
    # print(dftimeMinusCat.head())

    dftimeR=df.groupby(['category']).filter(lambda x: x['category'].value_counts() > 145)
    print(dftimeR.head())

    # dftimeR1=dftimeR.groupby(['category']).filter(lambda x: x['category'].value_counts() < 55)
    # print(dftimeR.head())


    # dfTimeUniqCat=pd.unique(df.category.ravel())
    # logging.info('unique category - ' + str(len(dfTimeUniqCat)))

    dfTimeUniqCatR=pd.unique(dftimeR.category.ravel())
    logging.info('count of unique category > 145  - ' + str(len(dfTimeUniqCatR)))

    logging.info('lenght')
    print(dftimeR.count())

    return dftimeR

def create_df (corpus_path) :
    # dfMinusCat = pd.read_pickle(dfMinusCat_path)
    # dataf = pd.read_pickle(dataf_path)

    df = pd.DataFrame(columns=('id', 'content', 'category'))
    # dfList = dfMinusCat['category'].tolist()

    file_path = corpus_path
    context = ET.iterparse(file_path, events=("start", "end"))

    context = iter(context)
    on_page_tag = False
    count = 0
    row_id = ''
    row_category =''
    row_content=''
    text_read = True
    logging.info('started')
    mlist = []
    categoryCounter = {}
    largecat = {}
    for  event, elem in context:
        tag = elem.tag
        value = elem.text

        if event == 'start' :

            if tag == 'page' :
                on_page_tag = True

            elif tag == 'id' :
                if on_page_tag :
                    if count == 650000:
                        break;
                    count += 1
                    if value is not None:
                        row_id = str(value)
                        if count % 5000 == 0 :
                            logging.info('completed ' + str(count))

            elif tag == 'text' :
                if on_page_tag :
                    if value is not None:
                        row_content = str(value)

            elif tag == 'category' :
                if on_page_tag :
                    if value is not None:
                        row_category = str(value)
                        if ((row_id != '') & (row_content != '')) :
                            # rowt = dict(zip(['id', 'content', 'category'], [row_id, row_content, row_category]))
                            # mlist.append(rowt)


                            count1 = categoryCounter.get(row_category)
                            if count1 == None:
                                count1 = 1
                                rowt = dict(zip(['id', 'content', 'category'], [row_id, row_content, row_category]))
                                mlist.append(rowt)
                                categoryCounter.update({row_category : count1})

                                # допускаем что для построения тематической модели для каждой категории достаточно 50 документов
                            elif count1 <= 150 :
                                count1 += 1
                                rowt = dict(zip(['id', 'content', 'category'], [row_id, row_content, row_category]))
                                mlist.append(rowt)
                                categoryCounter.update({row_category : count1})
                            elif count1 > 150 :
                                largecat.update({row_category : count1})

        if event == 'end' :
            if tag == 'page' :
                on_page_tag = False
                row_id = row_category = row_content = ''

            elif tag == 'text' :
                if on_page_tag :
                    if value is not None:
                        row_content = str(value)


        elem.clear()
    df = pd.DataFrame(mlist)
    for k in largecat.keys() :
        print(k)
    print('len of all cat' + str(len(mlist)))
    print('len of large cat' + str(len(largecat)))

    return df

def pwt_matrix(len_corp, path_to_df) :
    data = pd.read_pickle(path_to_df)
    print(data.count())
    data_len = data[0:len_corp]
    print('data size\n' + str(data_len.count()))
    #
    category = data_len['category'].tolist()
    # textsCat = [[word for word in document.lower().split(', ')] for document in category]
    # dictCat = corpora.Dictionary(textsCat)
    # print(dictCat)
    # dictCat.save('Data/dictCat.dict')
    #
    content = data_len['content'].tolist()
    # textsCont = [[word for word in document.lower().split()] for document in content]
    # dictCont = corpora.Dictionary(textsCont)
    # print(dictCont)
    # dictCont.save('Data/dictCont.dict')

    # logging.info('tolist')
    #
    # dictCat = corpora.Dictionary(category)
    # dictContent = corpora.Dictionary(content)
    #
    # print(dictCat)
    # print(dictContent)
    #
    # dictCat.save('Data/category.dict')
    # dictContent.save('Data/category.dict')

    vectorizerContent = CountVectorizer (min_df=1, dtype=np.uint16)
    vectorizerCategory = CountVectorizer (min_df=1,dtype=np.uint16)
    logging.info('vectorizer')


    #X-pwd
    pwd = vectorizerContent.fit_transform(content)
    #Xcat - pdt
    pdt = vectorizerCategory.fit_transform(category)
    #Xdat = vectorizerDate.fit_transform(dataT)



    logging.info('transform')

    # selected_feature_names_Cont = np.asarray(vectorizerContent.get_feature_names())
    # selected_feature_names_Cat = np.asarray(vectorizerCategory.get_feature_names())

    logging.info('started')

    i = 0
    Pwt = csr_matrix( (pwd.shape[1], pdt.shape[1]), dtype='uint16' )
    # Pwt1 = 0
    while len_corp > i:
        pwdT = pwd[i].transpose()
        mul = pdt[i].multiply(pwdT)
        Pwt = Pwt + mul
        # Pwtarr = Pwt.toarray()

        # xcArr = pdt[i].toarray()
        # pwdArr = pwd[i].toarray().transpose()
        # Pwt1 += np.array(xcArr, dtype='uint16') * np.array(pwdArr, dtype='uint16')
        # if np.array_equal(Pwtarr,Pwt1) :
        #     print('True')
        # # print (i)
        i += 1
        if i % 100 == 0:
            logging.info('iterated ' + str(i) + ' elements')
    Pwtarr = Pwt.toarray()
    np.save('Data/Pwt.npy', Pwtarr)

def phi_matrix(path_to_pwt) :
    Pwt=np.load(path_to_pwt)
    i1 = 0
    Phi = np.arange(Pwt.size, dtype='float16').reshape(Pwt.sum(axis=1).size, Pwt[0].size)
    for el4 in Pwt.sum(axis=1):
        i2 = 0
        for el1 in Pwt[i1]:
            Phi[i1][i2]=float(el1)/Pwt[i1].sum()
            i2 += 1
        # print(Pwt[i1])
        i1 += 1
        if i1 % 5000 == 0 :
            logging.info(str(i1) + ' from ' + str(Pwt.sum(axis=1).size))
    np.save('Data/Phi.npy', Phi)
    print (Phi)

def stem(text) :
    stemmer_ = Stemmer.Stemmer('russian')
    stemArray = stemmer_.stemWords(text.split(" "))
    outStr = ''
    for el in stemArray:
        if el != "":
            outStr = outStr + ' ' + el
    return outStr

def predict(text) :
    re_clean = re.compile(r'[^\w]')
    finalStr = text.lower()
    finalStr = re_clean.sub(' ', finalStr)
    for word in stop_words:
        finalStr = finalStr.replace(" " + word + " ", " ")
    stemmedStr = stem(finalStr)
    logging.info('\nПредсказание категори для текста:\n\n' + text )
    # print(stemmedStr)
    data = pd.read_pickle('Data\dfMinusCat1.pkl')
    data_len = data[0:70000]
    content = data_len['content'].tolist()
    category = data_len['category'].tolist()
    vectorizerContent = CountVectorizer (min_df=1)
    vectorizerCategory = CountVectorizer (min_df=1)
    logging.info('vectorizer')
    pwd = vectorizerContent.fit_transform(content)
    pdt = vectorizerCategory.fit_transform(category)
    pwdT = pwd.transpose()
    # xxcat-pdtT
    pdtT = pdt.transpose()
    logging.info('transpose OK')
    selected_feature_names_Cont = np.asarray(vectorizerContent.get_feature_names())
    selected_feature_names_Cat = np.asarray(vectorizerCategory.get_feature_names())

    Pwt = np.load('Data/Pwt.npy')
    Phi = np.load('Data/Phi.npy')
    countDoc = len(content)

    newWord = vectorizerContent.transform([stemmedStr]).toarray()
    # print (newWord)
    predictMatrixW = []
    j = 0
    for inNew in np.nditer(newWord):
        if inNew > 0:
            wordInDoc = float(pwdT[j].sum())
            k = 0
            for inCat in Pwt[j]:
                if inCat > 0:
                    wordInCat = float(pdtT[k].sum())
                    el_data = {}
                    el_data[selected_feature_names_Cat[k]] = Phi[j][k] * (1 - wordInCat / countDoc) * (
                    1 - wordInDoc / countDoc)
                    predictMatrixW.append(dict(el_data))
                k += 1
        j += 1
    dfMW = DataFrame(predictMatrixW)

    # print (dfM.sum())
    dfSumMW = dfMW.sum()
    i = 0
    dfSumMW.sort_values(ascending=False, kind='quicksort', na_position='last', inplace=True)
    for el in dfSumMW.T.iteritems():
        print(str(i) + ') ' + str(el[0]) + ' = ' + str(el[1]))
        i = i + 1

if __name__ == "__main__":
    os.chdir('C:\mlPLSI-master')
    df = clean_category('Data\DF1.pkl')
    df.to_pickle('Data\dfMinusCat1.pkl')
    dfTimeUniqCat=pd.unique(df.category.ravel())
    logging.info('unique category - ' + str(len(dfTimeUniqCat)))
    for uni in dfTimeUniqCat:
    	print(uni)

    pwt_matrix(70000, 'Data\dfMinusCat1.pkl')

    phi_matrix('Data/Pwt.npy')

    predict('поэтов, работавших на территории России в современных границах (граждане и подданные находившихся на её территории в различные эпохи государств, кроме живших в таких государствах')
    predict('В библиотеке расскажут о тонкостях перевода произведений Иосифа Бродского с представителями семейства кошачьих, к которым поэт питал особую любовь.')

    # df = pd.read_pickle('Data\dfMinusCat1.pkl')
    # print(df.count())
    # df = create_df('corpus_lite.xml')
    # df.to_pickle('Data\DF1.pkl')



