from pymongo import MongoClient
import pymongo


def get_db():
    string = "mongodb+srv://Kayram:He5ri6sKlad@cluster0.y33bn.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
    client = MongoClient(string)
    return client["tasks"]
