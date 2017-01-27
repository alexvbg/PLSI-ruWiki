import Stemmer

finalStr = 'выборы президента россии'

stemmer_ = Stemmer.Stemmer('russian')
stemArray = stemmer_.stemWords(finalStr.split(" "))
outStr = ''
for el in stemArray :
    if el != "" :
        outStr = outStr + ' ' + el

print(outStr)