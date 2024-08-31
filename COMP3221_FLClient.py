import socket
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader
import struct
import os

class MCLR(nn.Module):
    """
    borrow from W6_Tutorial_Federated_Learning-Solution
    """
    def __init__(self):
        super(MCLR, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class UserAVG():
    """
    borrow from W6_Tutorial_Federated_Learning-Solution
    """
    def __init__(self, client_id, model, learning_rate, batch_size):

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = get_data(client_id)
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        if batch_size:
            print_save(f"The client uses Mini Batch GD training of size {batch_size}")
            self.trainloader = DataLoader(self.train_data, batch_size, shuffle=True)
            self.testloader = DataLoader(self.test_data, batch_size, shuffle = True)
        else:
            print_save(f"The client uses Batch GD training")
            self.trainloader = DataLoader(self.train_data, self.train_samples)
            self.testloader = DataLoader(self.test_data, self.test_samples)

        self.loss = nn.NLLLoss()

        self.model = copy.deepcopy(model)

        self.id = client_id

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
            
    def train(self, epochs):
        LOSS = 0
        self.model.train()
        batch_iterator = iter(self.trainloader)
        for epoch in range(1, epochs + 1):
            self.model.train()
            try:
                X, y = next(batch_iterator)
            except StopIteration:  # End of dataloader
                batch_iterator = iter(self.trainloader)
                X, y = next(batch_iterator)
            
            
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
        print_save(f"Training loss: {loss.data:.2f}")
        return loss.data.item()
    
    def test(self):
        
        self.model.eval()
        total_correct = 0
        total_samples = 0

        for batch_idx, (X, y) in enumerate(self.testloader):
            output = self.model(X)
            preds = torch.argmax(output, dim=1)
            correct = torch.sum(preds == y).item()
            total_correct += correct
            total_samples += len(y)

        test_acc = total_correct / total_samples
        print_save(f"Testing accuracy of client: {test_acc * 100:.0f}%")
        return test_acc
    
def send_handshake(client_socket, client_id, data_size):
    """
    send a handshake protocol json string in bytes
    """
    handshake_data = {
        'id': client_id,
        'data_size': data_size
    }
    handshake_message = json.dumps(handshake_data)
    client_socket.sendall(handshake_message.encode('utf-8'))

def send_local_model(client_socket, client_id, user):
    """
    send a local model data message and its expected size
    """
    model_params = [param.data for param in list(user.model.parameters())]
    local_model_data_message = json.dumps([p.tolist() for p in model_params] )
    message_bytes = local_model_data_message.encode('utf-8')
    client_socket.sendall(struct.pack('!I',len(message_bytes)))
    client_socket.sendall(message_bytes)

def print_save(str):
    """
    output prompts to the terminal and save them to a file.
    """
    print(str)
    fp.write(str+'\n')

def get_data(client_id):
    """
    borrow from 'W6_Tutorial_Federated_Learning-Solution'
    and modified a bit
    """
    
    train_data = {}
    test_data = {}

    with open(os.path.join("FLdata","train",f"mnist_train_{client_id}.json"), "r") as f_train:
        train = json.load(f_train)
        train_data.update(train['user_data'])
    with open(os.path.join("FLdata","test",f"mnist_test_{client_id}.json"), "r") as f_test:
        test = json.load(f_test)
        test_data.update(test['user_data'])
    X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0']['y']
    X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.int64)
    train_samples, test_samples = len(y_train), len(y_test)
    return X_train, y_train, X_test, y_test, train_samples, test_samples

def main(client_id, port_client, opt_method):
    """
    file to save
    """
    global fp
    if opt_method:
        fp = open(f"{client_id}_batch_size_{opt_method}_log.txt", 'w')
    else:
        fp = open(f"{client_id}_gd_log.txt", 'w') 

    """
    initialization
    """
    user = UserAVG(client_id, model = MCLR(), learning_rate = 0.01, batch_size= opt_method)
    loss_log = []
    acc_log = []
    server_ip = 'localhost'
    server_port = 6000

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect((server_ip, server_port))
    print(f"{client_id} Connected to the server")

    #handshake
    send_handshake(server_socket,client_id, user.train_samples)

    #wait for model exchange
    while True:
        #receive a message
        size_message = server_socket.recv(4)
        inc_size = struct.unpack('!I',size_message)[0]
        remaining_size = inc_size
        params_json_bytes = b""
        while remaining_size != 0:
            params_json_bytes += server_socket.recv(remaining_size)
            remaining_size = inc_size - len(params_json_bytes)
        params_json = params_json_bytes.decode('utf-8')

        #close client
        if params_json == 'SHUTDOWN':
            print_save("----------------------------")
            print_save("Received shut down signal from the server")
            wrapper = {'loss_log':loss_log, 'acc_log':acc_log}
            final_data_bytes = json.dumps(wrapper).encode('utf-8')
            server_socket.sendall(struct.pack('!I',len(final_data_bytes)))
            server_socket.sendall(final_data_bytes)
            break

        #receive the global model
        tensor_params = [torch.tensor(p) for p in json.loads(params_json)]
        with torch.no_grad():
            for i, param in enumerate(list(user.model.parameters())):
                param.data.copy_(tensor_params[i])

        #training and feedbacks
        print_save("----------------------------")
        print_save(f"I am client {client_id[-1]}")
        print_save(f"receiving new global model")
        print_save("Local training...")
        loss_log.append(user.train(epochs = 2))
        acc_log.append(user.test())
        print_save("Sending new local model")
        send_local_model(server_socket,client_id,user)
    
    server_socket.close()
    fp.close()
    print("The client has been shut down")

if __name__ == "__main__":
    client_id = sys.argv[1]
    port_client = int(sys.argv[2])
    opt_method = int(sys.argv[3])
    main(client_id, port_client, opt_method)
