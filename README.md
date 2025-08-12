# Running Locally

1. Don't forget to get the correct credentials from aws.
2. Create a virtual environment and download the requirements. (Note: add --ignore-installed to it if there's an error in already installed packages)
3. `streamlit run app.py` or `py -m streamlit run app.py`.

# Running on EC2
1. `git clone` this repo
2. `nohup streamlit run app.py --server.port 80 --server.address 0.0.0.0`
