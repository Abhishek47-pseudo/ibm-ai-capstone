import os

from application.model import model_train, model_load

def main():
    data_dir = os.path.join("data","cs-train")
    
    ## train the model
    model_train(data_dir, test=False)

    ## load the model
    model = model_load()
    
    print("model training complete.")


if __name__ == "__main__":
    main()
