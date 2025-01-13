from pymongo import MongoClient

client = MongoClient('mongodb+srv://caljshim:cjs18745@cluster0.nlgb1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')

db = client['nba_database']
collection = db['players']