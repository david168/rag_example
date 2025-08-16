from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set Google API key 
# Change environment variable name from "GOOGLE_API_KEY" to the name given in 
# your .env file.
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def main():
    # Get embedding for a word.
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()
