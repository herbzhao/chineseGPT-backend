import os

from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from parameters import ATLAS_URL

load_dotenv()
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")


# Send a ping to confirm a successful connection
def get_client():
    mongoDB_username = os.getenv("MONGODB_USERNAME")
    mongoDB_password = os.getenv("MONGODB_PASSWORD")
    uri = f"mongodb+srv://{mongoDB_username}:{mongoDB_password}@{ATLAS_URL}"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi("1"))
    return client


def check_connection(client):
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)


def get_users_collection(client):
    # Create a new client and connect to the server
    db = client["GPTian"]
    users_collection = db["users"]
    return users_collection
