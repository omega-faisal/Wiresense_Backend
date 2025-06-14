import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random, os, numpy as np

df = pd.read_csv('predictions/ml/Final_Anomaly_Removed_Data.csv')

# Normalization function in range [0, 1]
def normalize_column(column,min_val,max_val):
    return (column - min_val) / (max_val - min_val)

# Inverse normalization function
def inverse_normalize_column(norm_column, min_val, max_val):
    return norm_column * (max_val - min_val) + min_val

# Configuration class
class Config:
    DEVICE = "cpu"
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/val"
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-6
    LAMBDA_CYCLE = 12.625839019572867
    NUM_EPOCHS = 100
    LOAD_MODEL = False
    SAVE_MODEL = True
    CHECKPOINT_GEN_INPUT = "gen_input.pth.tar"
    CHECKPOINT_GEN_OUTPUT = "gen_output.pth.tar"
    CHECKPOINT_DISC_INPUT = "disc_input.pth.tar"
    CHECKPOINT_DISC_OUTPUT = "disc_output.pth.tar"
    DROPOUT_RATE = 0.0
    MEAN = 0.0
    STD = 1.0
    PRINT_FREQ = 100
    SAVE_FREQ = 1

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.InstanceNorm1d(128),  # Batch normalization for stable training
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.InstanceNorm1d(256),  # Batch normalization for stable training
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.InstanceNorm1d(128),  # Batch normalization for stable training
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, output_dim)
        )
        # Skip connection for better gradient flow

    def forward(self, x):
        return self.model(x)

# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.InstanceNorm1d(128),  # Batch normalization for stable training
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(128, 64),
            nn.InstanceNorm1d(64),  # Batch normalization for stable training
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # Single node for real/fake classification
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
def load_checkpoint(checkpoint_file, model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file,map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint["state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer"])

model_save_path = { 
    'gen_input': 'predictions/ml/gen_input.pth.tar', 
    'gen_output': 'predictions/ml/gen_output.pth.tar', 
    'disc_input': 'predictions/ml/disc_input.pth.tar', 
    'disc_output': 'predictions/ml/disc_output.pth.tar'
}

# Define your models
input_dim, output_dim = 16, 3

disc_input = Discriminator(input_dim=input_dim).to(Config.DEVICE)
disc_output = Discriminator(input_dim=output_dim).to(Config.DEVICE)
gen_input = Generator(input_dim=output_dim, output_dim=input_dim).to(Config.DEVICE)
gen_output = Generator(input_dim=input_dim, output_dim=output_dim).to(Config.DEVICE)

# Define your optimizers
opt_disc = optim.Adam(
    list(disc_input.parameters()) + list(disc_output.parameters()),
    lr=Config.LEARNING_RATE, betas=(0.5, 0.999)
)
opt_gen = optim.Adam(
    list(gen_input.parameters()) + list(gen_output.parameters()),
    lr=Config.LEARNING_RATE, betas=(0.5, 0.999)
)

# Load checkpoints
# Load checkpoints
load_checkpoint(model_save_path['gen_input'], gen_input, opt_gen)
load_checkpoint(model_save_path['gen_output'], gen_output, opt_gen)
load_checkpoint(model_save_path['disc_input'], disc_input, opt_disc)
load_checkpoint(model_save_path['disc_output'], disc_output, opt_disc)

# Now your models and optimizers are loaded and ready for use
print("Models and optimizers loaded successfully.")

df.drop(columns='Unnamed: 0',inplace=True)
df.drop(columns = ['%SI','%FE','%TI','%V','%AL'],inplace = True)

# Specify the columns you're interested in
columns = ['   UTS', 'Elongation', 'Conductivity']
outputs = ['UTS', 'Elongation', 'Conductivity']
other_columns = [col for col in df.columns if col not in columns]

import torch
import pandas as pd
import json

def reverse_predict(json_input):
    """
    Function to predict based on input JSON and return denormalized results in JSON format.

    Parameters:
    - json_input (str): JSON string containing the input data.
    - df (pd.DataFrame): DataFrame containing the original data for normalization.
    - outputs (list): List of expected numerical output keys.
    - gen_input (callable): Function or model to generate predictions.

    Returns:
    - str: JSON string containing the denormalized prediction results.
    """
    # Parse the JSON input
    input_data = json.loads(json_input)

    # Extract numerical fields from the input and convert to a Series
    input_numerical = {key: float(value) for key, value in input_data.items() if key in outputs}

    # Create a dictionary to store normalized input values
    normalized_values = {}

    for key, value in input_numerical.items():
        # Adjust column name if necessary
        col = key
        if key == 'UTS':
            col = '   UTS'  # Handle specific column name formatting if required

        # Get the min and max values for the column
        min_val = df[col].min()
        max_val = df[col].max()

        # Normalize the value
        norm_val = normalize_column(value, min_val, max_val)

        # Store normalized value
        normalized_values[col] = norm_val

    # Convert normalized values to a 2D tensor
    first_row_tensor = torch.tensor(list(normalized_values.values()), dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)

    # Pass the reshaped tensor to the model
    predictions_output = gen_input(first_row_tensor)

    # Denormalize each column of the tensor
    denormalized_results = {}
    for i, column in enumerate(predictions_output[0]):
        column_name = other_columns[i]  # Get the column name (assuming df contains the original data)
        min_val = df[column_name].min()  # Get the min value for the column
        max_val = df[column_name].max()  # Get the max value for the column

        # Denormalize the value 
        denormalized_value = inverse_normalize_column(column.detach().item(), min_val, max_val)

        # Add the result to the dictionary
        denormalized_results[column_name] = round(denormalized_value, 5)  # Limit to 5 decimal places

    # Convert the results to JSON and return
    return json.dumps(denormalized_results, indent=2)