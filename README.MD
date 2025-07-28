# RAG Chatbot for Used Car Inquiries and Recommendations

This project is a Retrieval-Augmented Generation (RAG) AI chatbot designed to assist users in purchasing secondhand cars. Leveraging a Large Language Model (LLM) and enhanced by real-time retrieval from a curated dataset of used car discussions and user reviews from Reddit, this chatbot provides contextually-relevant answers based on real and lived experiences. 

__Goal:__ Develop an AI-driven chatbot that delivers actionable purchasing advice and insights on used cars by utilizing a local knowledge base constructed from real user discussions and experiences. I aim to provide an easier way for first-time car buyers to make informed decisions and save time on tedious manual research.

__Objectives:__
1. _Data Pipeline Implementation:_
Design and build a robust ETL pipeline to extract, clean, and structure used car discussions and reviews from the Reddit API into a local database.
2. _Vector Database Integration:_
Construct a scalable vector database for efficient storage and retrieval of text embeddings derived from the curated dataset.
3. _Semantic Search & Retrieval:_
Develop a retriever module that embeds user queries, performs semantic similarity searches against the vector database, and surfaces the most relevant documents.
4. _RAG Pipeline Integration:_
Integrate the retrieval process with prompt augmentation and an LLM API to generate high-quality, context-aware responses based on both model knowledge and retrieved information.
5. _[Optional] User Interface Design:_
Develop an intuitive user interface to facilitate natural language inquiries and clearly present chatbot recommendations and supporting references.

__Motivation:__ As a first-time car buyer, it's easy to get lost in the sea of options in the used car market. The plethora of options for brands, models, model years and trims, and whatnot, can make it daunting to do your own research. Personally, I find myself gravitating to online forums when doing my research on used cars as I believe in the concept of "Wisdom of the crowd", wherein majority sentiment may be indicative of some measurable and verifiable phenomena. 

In the case of used cars, if a significant number of people praise a specific Japanese-manufactured vehicle model while citing its faults, an algorithm that sufficiently captures this data may be able to present a high confidence level for the said model while presenting an overview of common issues to be aware of. In doing so, the work involved in research is cut in half, and prospective buyers can trim their options down to a few models that fit their criteria, while being made aware of the salient points that require extensive due diligence (e.g. common issues, upkeep, regulations) that only humans can do.

## How to run this program:
1. Configure the .env.template file under the config directory with your client id and client secret keys from Reddit API and chosen LLM keys. Rename to .env after populating the fields.
2. Run the etl.py script to extract data from Reddit and update the datasets in the data directory. Otherwise, you can skip this step and utilize the pre-scraped data under the data directory.
3. Run the app.py script to access the CLI chatbot. It will prompt you to build a new vector store when executed. This will create a persistent ChromaDB collection inside the project root containing the most recent documents from the local parquet files. In subsequent re-runs of the app, you can skip building the vector store for faster startup.
4. Prompt the chatbot with your car purchasing inquiries.