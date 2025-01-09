import os
import pickle


def save_q_table(q_table, history, file_name="q_table.pkl"):
    with open(file_name, "wb") as f:
        pickle.dump((q_table, history), f)

def load_q_table(file_name="q_table.pkl"):
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            return pickle.load(f)
    return {}, []
