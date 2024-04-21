from kaggle.api.kaggle_api_extended import KaggleApi


api = KaggleApi()
api.authenticate()

def submit_to_kaggle(sub_csv_path , message = ""):
    sub_csv_path = str(sub_csv_path)
    if message == "":
        message = sub_csv_path[sub_csv_path.rfind("/"):]
    
    result = api.competition_submit(sub_csv_path,
                        message,
                        'vinbigdata-chest-xray-abnormalities-detection')

    return result

    