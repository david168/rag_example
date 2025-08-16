# Langchain RAG Tutorial

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```bash
     conda install onnxruntime -c conda-forge
    ```
    See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additonal help if needed. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.


2. Now run this command to install dependenies in the `requirements.txt` file. 

```bash
pip install -r requirements.txt
```

3. Install markdown dependencies with: 

```bash
pip install "unstructured[md]"
```

**Note**: If you encounter dependency conflicts during installation, try removing version constraints from the requirements.txt file to let pip automatically resolve compatible versions.

## Set up Google API Key

You'll need to set up a Google Cloud account and get an API key for the Gemini API.

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create an API key
3. Create a `.env` file in the project root with your API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Create database

Create the Chroma DB.

```bash
python create_database.py
```

## Query the database

Query the Chroma DB.

```bash
python query_data.py "How does Alice meet the Mad Hatter?"
```

## Recent Improvements

We've made several improvements to enhance the RAG system's performance:

1. **Upgraded the embedding model** from `models/embedding-001` to `models/text-embedding-004`, which provides better semantic similarity matching.

2. **Increased chunk size** from 500 to 800 characters with 200 character overlap, which includes more context in each chunk.

3. **Increased the number of retrieved chunks** from 3 to 10, which increases the likelihood of retrieving the relevant chunk.

With these improvements, the system now correctly answers the question "How does Alice meet the Mad Hatter?" with:

"Alice meets the Mad Hatter at a tea party under a tree in front of the March Hare's house. The Mad Hatter and March Hare are already having tea with a sleeping Dormouse when Alice arrives."

This is a significant improvement over previous responses that only mentioned Alice meeting the Hatter after following the Cat's directions.

### How Increasing Retrieved Chunks Improves Accuracy

When we increased the number of retrieved chunks from 3 to 10, it significantly improved accuracy for this specific query because:

1. **Higher Chance of Retrieving Relevant Content**: With only 3 chunks, we were missing the key chunk that contains the description of Alice's arrival at the tea party. The key chunk (which describes Alice coming to the table and the "No room!" exchange) was ranked 8th in the similarity search results. By retrieving 10 chunks instead of 3, we ensured that this important chunk was included in the context.

2. **More Complete Context for the LLM**: With more chunks, the LLM has access to a broader range of information from the source document. It can see not just isolated quotes but the narrative flow and connections between different parts of the story. This helps the LLM provide more comprehensive and accurate answers.

3. **Better Handling of Complex Queries**: Questions like "How does Alice meet the Mad Hatter?" require information from different parts of the text:
   - The Cheshire Cat's directions
   - Alice's journey to the location
   - The actual meeting/encounter
   Retrieving more chunks increases the likelihood of capturing all these elements.

4. **Redundancy and Confirmation**: Having more chunks provides redundancy - if one chunk has relevant information, others might have complementary details. The LLM can cross-reference information across multiple chunks to form a more accurate understanding.

The trade-off of retrieving more chunks is increased processing time and token usage, but in our case, the improvement in accuracy was worth this trade-off.

Here is a step-by-step tutorial video: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).
