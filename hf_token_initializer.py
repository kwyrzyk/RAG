from huggingface_hub import login   


def init_hf_token(file_path):
    with open(file_path, 'r') as file:
        token = file.read()
        login(token)