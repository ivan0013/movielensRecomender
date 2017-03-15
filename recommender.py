import numpy as np
import pandas as pd
import math
import operator
import sys
import json

myUser = {}

TRESHOLD_USERUSER = 50
MINIMUM_COMMON_MOVIES = 3
   
def productoVectorial(a,b):
    count = 0
    for i in range(0, len(a)):
        count += (a[i]*b[i])
    
    return count

def norma(a):
    count = 0
    
    for e in a:
        count += (e*e)
    
    return math.sqrt(count)

def media(a):
    res = 0
    
    if len(a) > 0:
        res = float(sum(d for d in a)) / len(a)
    
    return res

def restaCuadrados(a, media):
    total = 0
    
    for v in a:
        total = total + ((v - media) ** 2)
        
    return total
        

def pearson(u1, u2, u1m, u2m):  
    u1 = sorted(u1.items(), key=operator.itemgetter(0))
    u2 = sorted(u2.items(), key=operator.itemgetter(0))
    
    u1 = [x[1] for x in u1]
    u2 = [x[1] for x in u2]
    
    numerador = 0
    k = 0
    for v in u1:        
        numerador = numerador + ((v-u1m)*(u2[k]-u2m))
        k+=1
        
    denominador = math.sqrt(restaCuadrados(u1, u1m)*restaCuadrados(u2, u2m))
    
    try:
        coef = numerador/denominador
    except:
        coef = 0
        
    return coef

def cosine(u1, u2):
 
    u1 = sorted(u1.items(), key=operator.itemgetter(0))
    u2 = sorted(u2.items(), key=operator.itemgetter(0))
    
    u1 = [x[1] for x in u1]
    u2 = [x[1] for x in u2]
    
    res = 0
    
    try:
        res = productoVectorial(u1, u2)/(norma(u1) * norma(u2))
    except:
        res = 0
    return res

def weightAvg(users, ratings, simils, movie_id):
    res = 0
    myAvg = float(sum(myUser.values())) / float(len(myUser.keys()))
    usedUser = list()
    numerador = 0
    denominador = 0
    
    for t in simils:
        user, simil = t
        voto = ratings[(ratings['user_id']==user) & (ratings['movie_id'] == movie_id) ]
        
        try:
            voto = float(voto['rating'].tolist()[0])
        except:
            voto = 0
        
        if voto != 0 and len(usedUser) < TRESHOLD_USERUSER:
            media = users[users['user_id'] == user]['avg'].tolist()[0]
            numerador = numerador + (voto - media) * simil
            usedUser.append(user)
                
    for user in usedUser:
        denominador = denominador + float([item for item in simils if item[0] == user][0][1])
    
    try:
        res = myAvg + (float(numerador)/float(denominador))
    except:
        res = 0
    
    if res > 5 : res = 5
    if res > 0 and res < 1 : res = 1
    if res < 0 : res = 0
    
    return res    

def getSimilarUsers(myUser):
# Reading users file:
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
    encoding='latin-1')
   
# Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
    encoding='latin-1')

# Reading items file:
    i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
    'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
    
    mySimilarUsers = {}
    for user in users.user_id.unique():
         rats = ratings.loc[ratings['user_id'] == user]['rating'].tolist()
         films = ratings.loc[ratings['user_id'] == user]['movie_id'].tolist()
         mySimilarUsers[user] = {key:value for key, value in zip(films,rats)}
         
     # Get the vote average for each user
    users['avg'] = 0 
    for user in ratings.user_id.unique():
        numVotos = len(ratings.loc[ratings['user_id'] == user]) 
        rats = ratings.loc[ratings['user_id'] == user]['rating'].tolist()
        users.ix[users["user_id"] == user, 'avg'] = (float(sum(rats))/float(numVotos))

    
    simils = {}
    u2m = sum(myUser.values())/len(myUser.values())
    for user, ratings in mySimilarUsers.iteritems():
        u1 = {}
        u2 = {}
        
        for key, rating in ratings.iteritems():
            if key in myUser: 
                u1[key] = rating
        for key, rating in myUser.iteritems():
            if key in ratings: 
                u2[key] = rating 
        
        if bool(u1) and bool(u2) and len(u1) >= MINIMUM_COMMON_MOVIES: 
            u1m = users.loc[users['user_id'] == user]['avg'].tolist()[0]
            simils[user] = pearson(u1, u2, u1m, u2m)
        else:
            simils[user] = 0

    return simils
        
def getRecommendedMovies(myUser, tipo):
    print("This process is going to take several minutes. Please wait")
    
# Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

# Reading items file:
    i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
    'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
    
    totalMovies = str(len(ratings.movie_id.unique().tolist()))
    
    if tipo == 1: #useruser 
        # Reading users file:
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
        
        # Get the vote average for each user
        users['avg'] = 0 
        for user in ratings.user_id.unique():
            numVotos = len(ratings.loc[ratings['user_id'] == user]) 
            rats = ratings.loc[ratings['user_id'] == user]['rating'].tolist()
            users.ix[users["user_id"] == user, 'avg'] = (float(sum(rats))/float(numVotos))
        
        # Get SimilarUsers and sorting
        simils = getSimilarUsers(myUser)
        similsAux = sorted(simils.items(), key=operator.itemgetter(0))
        simils = sorted(similsAux, key=lambda tup: tup[1], reverse = True)
        
        
        # Get the forecast    
        items["forecast"] = float(0)
        moviesAnalized = 0
        for movie_id in ratings.movie_id.unique():
            if movie_id not in myUser.keys():
                vote = weightAvg(users, ratings, simils, movie_id)
                items.set_value(movie_id-1, 'forecast', vote)
                moviesAnalized += 1
                sys.stdout.write('Processed %s / %s movies \r' % (str(moviesAnalized),totalMovies))
                sys.stdout.flush()         
         
        recommended = {}
        movie = items['movie_id'].tolist()
        title = items['movie_title'].tolist()
        forecast = items['forecast'].tolist()
        
        recommended = zip(movie, title, forecast)
        
    return recommended
                
        
    
    
    

def cero():
    print("Good bye.")
    
def uno():
    print("** To know which film corresponds to each id you should check the movielens 100k file. **")
    print("** Any rating above 5 and below 1 will be set to 5 or 1. **")
    print("** If you want to stop rating introduce 0 as the movie id. Id's above 1682 are not allowed. **")
    
    salir = False
    while(salir == False):
        try:
            movie_id = int(raw_input("The id of the movie you want to rate: "))
        except:
            movie_id = 0
            print("Unrecognized option. Leaving ...")
            
        
        if(movie_id < 1 or movie_id > 1682):
            movie_id = 0
        
        if(movie_id != 0):
            try:
                rate = int(raw_input("Rating for "+str(movie_id)+" is: "))
            except:
                rate = 3
                print("Unrecognized option. Using 3 as rating.")
                
            if rate > 5:
                rate = 5
            if rate < 1:
                rate = 1
            myUser[movie_id] = rate
        else:
            salir = True
    print myUser
    
def dos():
    
    if(not myUser or len(myUser.values()) < MINIMUM_COMMON_MOVIES):
        print("Before this step, you have to rate "+str(MINIMUM_COMMON_MOVIES)+" movies.\n");
    else:
        try:
            userAmount = int(raw_input("Specify number of user to retrieve: "))
        except:
            userAmount = 5
            print("Unrecognized option. Using 5 as the amount to retrieve.")

        print "The users who are more likely to you are:"
        
        simils = getSimilarUsers(myUser)
        
        similsSorted = sorted(simils, key=simils.get, reverse=True)[:userAmount]
        print similsSorted
    
def tres():
    if(not myUser or len(myUser.values()) < MINIMUM_COMMON_MOVIES):
        print("Before this step, you have to rate "+str(MINIMUM_COMMON_MOVIES)+" movies.\n");
    else:
        try:
            movieAmount = int(raw_input("Specify number of movies to retrieve: "))
        except:
            movieAmount = 5
            print("Unrecognized option. Using 5 as the amount to retrieve.")

        print "Your top "+str(movieAmount)+" recommended movies are:"
        
        simils = getRecommendedMovies(myUser, 1)
        
        try:
            recommended = sorted(simils, key=lambda tup: tup[2], reverse = True)
        except:
            recommended = {}
            print("We're sorry. We have not enough data to make a prediction based on your ratings")
             
             
        for rec in recommended[:movieAmount]:
            movie_id, title, rate = rec
            print(title + ": "+str(rate))


def main():

    options = {
        0:cero,
        1:uno,
        2:dos,
        3:tres
    }
    
    res = None;
    while(res !=0 ):
        print("Wellcome to the recommeder, you can choose one of the following options:")
        print("  1) Rate your films (ratings are erased when you left the application)")
        print("  2) Get users that you can match with")
        print("  3) Get a list of recommended films")
        print("  0) Exit")
        print("\n")
        
        try:
            res = int(raw_input("Your choice: "))
        except:
            res = 0
            print("Unrecognized option. Leaving ...")
        
        if res != 0 and res != 1 and res != 2 and res != 3:
            res = 0
            
        options[res]()
        print("\n\n")
        print("****************************************************************************************")
        print("\n")
            


            

    
    
    
    
    
main();
