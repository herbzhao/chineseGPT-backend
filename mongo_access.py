import os

from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")


mongoDB_username = os.getenv("MONGODB_USERNAME")
mongoDB_password = os.getenv("MONGODB_PASSWORD")
uri = f"mongodb+srv://{mongoDB_username}:{mongoDB_password}@cluster0.wh9yjkq.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi("1"))

# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
