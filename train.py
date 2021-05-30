import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model import LSTMFakeClassifier

def model_fn(model_dir):
    
    '''
    This model_fn is a necessary component for deploying an estimator as a predictor.
    It seeks model_fn and loads in the model previously fit by the estimator. This model
    and the supporting word_dict should be saved in the folder model_dir.
    '''
    
    print("Loading model.")

    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))
    
    print('Setting up device (CUDA by default if available, else CPU')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #extracting all the necessary model parameters from the saved file.
    embedding_dim = model_info['embedding_dim']
    hidden_dim = model_info['hidden_dim']
    vocab_size = model_info['vocab_size']
    num_layers = model_info['num_layers']
    dropout = model_info['dropout']
    
    #insert collected model parameters into model
    model = LSTMFakeClassifier(embedding_dim, hidden_dim, vocab_size, num_layers, dropout)

    #load model state dictionary as well
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    print('Loading word_dict.')
    
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)
    
    model.to(device).eval()

    print("Model loaded.")
    return model

def fetch_trainloader(batch_size, train_dir):
    
    ###read in as Pandas DataFrame with no headers or names to ensure correct formatting.
    train_data = pd.read_csv(os.path.join(train_dir, "train.csv"), header=None, names=None)

    ###take data, convert to torch tensors
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()   ###classes 
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long() ###feature dataset

    ###combine class and features into one TensorDataset for SageMaker
    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device):
    
    '''
    Fetches batches from a given train_loader.
    Implements model.
    Runs epochs and measures total cumulative loss using a specific loss function (loss_fn).
    '''
    criterion = loss_fn
    
    for epoch in range(1, epochs + 1):
        
        model.train()
        total_loss = 0
        
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            logps  = model.forward(batch_X)
            loss = criterion(logps, batch_y)
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.data.item()
            
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))
        
def parser_arguments(parser):
    
    '''
    Takes an argparse.ArgumentParser() object
    Updates the parser with the necessary arguments for training and SageMaker, as well as the model hyperparameters.
    Returns the parse_args() object for usage in training.
    '''

    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='NE',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='NH',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='NV',
                        help='size of the vocabulary (default: 5000)')
    parser.add_argument('--num_layers', type=int, default=1, metavar='NL',
                        help='number of RNN layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0., metavar='DO',
                        help='likelihood of layer dropout (default:0.0). cannot be > 0 unless num_layers > 1.')


    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    
    args = parser_arguments(argparse.ArgumentParser())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed) ###a fixed seed to ensure consistency of results when reporting and testing

    train_loader = fetch_trainloader(args.batch_size, args.data_dir)
    model = LSTMFakeClassifier(args.embedding_dim, args.hidden_dim, 
                               args.vocab_size, args.num_layers, args.dropout).to(device)

    with open(os.path.join(args.data_dir, "word_dict.pkl"), "rb") as f:
        model.word_dict = pickle.load(f)

    print('''
            Model loaded with the following paramters:\n
            embedding_dim {:.0f}\n 
            hidden_dim {:.0f}\n
            vocab_size {:.0f}\n
            num_layers {:.0f}\n
            dropout {:.2f}
          '''.format(args.embedding_dim, args.hidden_dim, args.vocab_size, 
                       args.num_layers, args.dropout)
         )

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model for future use 
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
        torch.save(model_info, f)


    # Save the model itself
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
    
    # Save the word dict -- this is only used for the predictor (which needs a model_fn)
    word_path = os.path.join(args.model_dir, 'word_dict.pkl')
    with open(word_path, 'wb') as f:
        torch.save(model.word_dict, f)
