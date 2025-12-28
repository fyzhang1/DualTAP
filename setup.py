import os


def setup_project():
    directories = [
        "data",
        "checkpoints",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"create directory: {directory}/")

if __name__ == "__main__":
    setup_project()

