# discord_chatbot

This project implements a Retrieval-Augmented Generation (RAG) based Discord chatbot designed to reponse general information queries.

It leverages [Qdrant](https://qdrant.tech/) as a high-performance vector database, the `text-embedding-3-small` model for efficient embedding generation, and `gpt-4o-mini` as the primary language model for response generation.

Additionally, the system includes a search step that extracts potential usernames from user input, allowing for targeted searches against a database and further improving the quality and relevance of its responses.

## requirement

A list of environment variables are required:

- `DISCORD_TOKEN`: discord token for the bot
- `OPENAI_API_KEY`: openai api token
- `QDRANT_URL`: url to the Qdrant cluster
- `QDRANT_API_KEY`: Qdrant api token
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, and `AWS_SESSION_TOKEN`(optional if temporary credentials is used): AWS access
- `DYNAMODB_TABLE`: Dynamodb table name
