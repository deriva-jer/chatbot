# Running Locally

1. Don't forget to get the correct credentials from aws.
2. Create a virtual environment and download the requirements. (Note: add --ignore-installed to it if there's an error in already installed packages)
3. `streamlit run app.py` or `py -m streamlit run app.py`.

# Running on EC2
1. `git clone` this repo
2. `nohup streamlit run app.py --server.port 80 --server.address 0.0.0.0`
3. To stop it, `sudo lsof -i :80`, and then `sudo kill -9 [the PID]`

---

**Some more notes:**

For AWS inquiry: Sandeep

Finance: Gavin Humphrey

S3 - data \
Bedrock -> knowledge base - index \
DynamoDB -> memory/database \
Python script -> Streamlit app \
EC2 -> Instances - server - host the app

---

**Deployment videos**
- https://www.youtube.com/watch?v=oynd7Xv2i9Y
- https://www.youtube.com/watch?v=DflWqmppOAg

**Creating history langchain with dynamodb**
- https://python.langchain.com/v0.2/docs/integrations/memory/aws_dynamodb/
- https://aws.amazon.com/blogs/database/build-a-scalable-context-aware-chatbot-with-amazon-dynamodb-amazon-bedrock-and-langchain/
