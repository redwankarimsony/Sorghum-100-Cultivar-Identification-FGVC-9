import os
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


from config import CFG
from models_zoo import CustomEffNet
from models_zoo import LitSorghum
from dataset import SorghumDataset, get_transform


def make_predictions(model_name, ckpt_file, data_dir, inp_sub_file, out_sub_file):
    # Loadinng the sample submission dataframe
    sub = pd.read_csv(inp_sub_file)
    # Loading the original Cultivar Names from the train file.
    train_df = pd.read_csv(inp_sub_file.replace('sample_submission.csv', 'train_cultivar_mapping.csv')).dropna()
    unique_cultivars = list(train_df["cultivar"].unique())

    # Modifying the sample submission dataframe to include the full path of the images
    sub["file_path"] = sub["filename"].apply(lambda image: os.path.join(data_dir, 'test',  image))
    sub["cultivar_index"] = 0
    print(sub.head())


    # Getting the data for inference
    test_dataset = SorghumDataset(sub,  get_transform('valid'))
    test_loader = DataLoader(test_dataset, 
                             batch_size=CFG.batch_size, 
                             shuffle=False, 
                             num_workers=CFG.num_cpu_workers)


    # Loading the model with last weights
    model = CustomEffNet(model_name=model_name, pretrained=False)
    model.load_state_dict(torch.load(ckpt_file)['state_dict'])
    model.cuda()
    model.eval()


    # Generating The Predictions
    predictions = []
    for batch in tqdm(test_loader):
        image = batch['image'].cuda()
        with torch.no_grad():
            outputs = model(image)
            preds = outputs.detach().cpu()
            predictions.append(preds.argmax(1))




    tmp = predictions[0]
    for i in range(len(predictions) - 1):
        tmp = torch.cat((tmp, predictions[i+1]))
    predictions = [unique_cultivars[pred] for pred in tmp]

    sub = pd.read_csv(inp_sub_file)
    sub["cultivar"] = predictions
    sub.to_csv(out_sub_file, index=False)

    print(f"Predictions are written in \n{out_sub_file}")



if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    model_name = "densenet121"
    
    for ver in range(1,6):
        version = f"version_{ver}"
        checkpoint_file = os.path.join("logs", model_name, version, "checkpoints", "best.ckpt")
        output_submission_file = os.path.join("results", f"{model_name}-{version}-submission.csv")
        input_submission_file = os.path.join(CFG.data_dir, 'sample_submission.csv')

        print(checkpoint_file, output_submission_file)

        make_predictions(model_name, checkpoint_file, CFG.data_dir, input_submission_file, output_submission_file)












