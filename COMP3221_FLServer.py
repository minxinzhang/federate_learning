import socket
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import threading
import random
import struct

class MCLR(nn.Module):

    def __init__(self):
        super(MCLR, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output
    
def aggregate_models(global_model, clients_info, sample_list):
    #reset the global model
    for param in global_model.parameters():
        param.data = torch.zeros_like(param.data)
        
    #use selected local models to integrate
    total_train_samples = 0
    for client_id in sample_list:
        total_train_samples += clients_info['sizes'][client_id]

    for client_id in sample_list:
        for server_param, user_param in zip(global_model.parameters(), clients_info['models'][client_id].parameters()):
            server_param.data = server_param.data + user_param.data.clone() * clients_info['sizes'][client_id] / total_train_samples
    return global_model

def send_global_model(clients_info, global_model,sample_list):
    """
    serialize the global model and send it
    """
    model_params = [param.data for param in list(global_model.parameters())]
    global_model_data_message = json.dumps([p.tolist() for p in model_params]).encode('utf-8')
    global_model_size = len(global_model_data_message)
    for client_id in sample_list:
        clients_info['connections'][client_id].sendall(struct.pack('!I',global_model_size)) 
        clients_info['connections'][client_id].sendall(global_model_data_message)
    
def receive_a_client_model(clients_info, client_id):
    """
    receive a local model from
    @client_id
    """
    #receive a message
    size_message = clients_info['connections'][client_id].recv(4)
    inc_size = struct.unpack('!I',size_message)[0]
    params_json_bytes = b""
    remaining_size = inc_size
    while remaining_size != 0:
        params_json_bytes += clients_info['connections'][client_id].recv(remaining_size)
        remaining_size = inc_size - len(params_json_bytes)
    params_json = params_json_bytes.decode('utf-8')
    
    #load to the local model
    tensor_params = [torch.tensor(p) for p in json.loads(params_json)]
    with torch.no_grad():
        for i, param in enumerate(list(clients_info['models'][client_id].parameters())):
            param.data.copy_(tensor_params[i])
    print(f"Getting local model from client {client_id}")

def handle_client(client_socket, clients_info):
    """
    reserve a data structure to store clients' information
    """
    clients = clients_info['clients']
    client_data_sizes = clients_info['sizes']
    client_models = clients_info['models']
    client_sockets = clients_info['connections']

    message = client_socket.recv(1024).decode('utf-8')
    data = json.loads(message)
    client_id = data['id']
    data_size = data['data_size']
    print(f"[SERVER] Received handshake from client {client_id[-1]} with data size {data_size}")

    clients.append(client_id)
    client_data_sizes[client_id] = data_size
    client_models[client_id] = nn.Linear(784,10)
    client_sockets[client_id] = client_socket

def new_client(server,clients_info,new_client_joined,stop_event):
    """
    establish connections from new devices after the initialization
    count down
    """
    print("[SERVER] still accepts new client handshakes")
    while not stop_event.is_set():
        try:
            client_socket, _ = server.accept()
            print("[SERVER] A new client joined")
            new_client_joined.append(client_socket)
        except socket.timeout:
            pass

def init_model_rand():
    """
    randomly initialize the global model
    """
    model = nn.Linear(784,10)
    nn.init.xavier_uniform_(model.weight)
    nn.init.xavier_uniform_(model.bias)
    return model

def prepare_server_socket(port):
    """
    create the socket listening to incoming connections
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('', port))
    server.listen(5)
    print(f"[SERVER] Listening on port {port}") 
    return server

def wait_for_first_handshake(server_socket,clients_info):
    """
    handles clients' handshaking messages
    """
    client_conn, _ = server_socket.accept()
    print("[SERVER] The first handshake message has been received. The server will wait for 30s")
    client_thread = threading.Thread(target=handle_client, args=(client_conn, clients_info))
    client_thread.start()
    return time.time()
    
def init_clients_info():
    """
    default client information data structure
    """
    clients = []
    client_data_sizes = {}
    client_models = {}
    client_connections = {}
    clients_info = {'clients':clients, 'sizes':client_data_sizes,
                    'models':client_models,'connections':client_connections}
    return clients_info

def receive_logs_nonsub(clients_info):
    """
    Non subsampling mode;
    retrieve local model evaluations and normalize
    them to be a log of average global model evaluation 
    per communication round.
    """
    loss_logs = {}
    acc_logs = {}
    for client_id in clients_info['clients']:
        size_message = clients_info['connections'][client_id].recv(4)
        inc_size = struct.unpack('!I',size_message)[0]
        params_json_bytes = b""
        remaining_size = inc_size
        while remaining_size != 0:
            params_json_bytes += clients_info['connections'][client_id].recv(remaining_size)
            remaining_size = inc_size - len(params_json_bytes)
        params_json = params_json_bytes.decode('utf-8')
        data_dict = json.loads(params_json)
        loss_logs[client_id] = (data_dict['loss_log'])
        acc_logs[client_id] = (data_dict['acc_log'])
    avg_loss = [0] * 100
    avg_acc = [0] * 100
    totol_sample = 0
    for client_id in clients_info['clients']:
        totol_sample += clients_info['sizes'][client_id]
        for i in range(len(loss_logs[client_id])):
            len_diff = 100 - len(loss_logs[client_id])
            avg_loss[i + len_diff] += loss_logs[client_id][i] * clients_info['sizes'][client_id]
            avg_acc[i + len_diff] += acc_logs[client_id][i] * clients_info['sizes'][client_id] 
    for i in range(100):
        avg_loss[i] /= totol_sample
        avg_acc[i] /= totol_sample
        avg_acc[i] *= 100
    return avg_loss, avg_acc

def receive_logs_sub(clients_info,sample_sequence):
    """
    Subsampling mode;
    retrieve local model evaluations and normalize
    them to be a log of average global model evaluation 
    per communication round.
    """
    loss_logs = {}
    acc_logs = {}
    
    for client_id in clients_info['clients']:
        size_message = clients_info['connections'][client_id].recv(4)
        inc_size = struct.unpack('!I',size_message)[0]
        params_json_bytes = b""
        remaining_size = inc_size
        while remaining_size != 0:
            params_json_bytes += clients_info['connections'][client_id].recv(remaining_size)
            remaining_size = inc_size - len(params_json_bytes)
        params_json = params_json_bytes.decode('utf-8')
        data_dict = json.loads(params_json)
        loss_logs[client_id] = (data_dict['loss_log'])
        acc_logs[client_id] = (data_dict['acc_log'])

    loss_iterators = {}
    acc_iterators = {}
    for client_id in clients_info['clients']:
        loss_iterators[client_id] = iter(loss_logs[client_id])
        acc_iterators[client_id] = iter(acc_logs[client_id])
    avg_loss = []
    avg_acc = []
    for sample_list in sample_sequence:
        totol_sample = 0
        loss_sum = 0
        acc_sum = 0
        for client_id in sample_list:
            totol_sample += clients_info['sizes'][client_id] 
            loss_sum += next(loss_iterators[client_id]) * clients_info['sizes'][client_id]
            acc_sum += next(acc_iterators[client_id]) * clients_info['sizes'][client_id]
        avg_loss.append(loss_sum/totol_sample)
        avg_acc.append(acc_sum/totol_sample*100)

    return avg_loss, avg_acc

def terminates_clients(clients_info):
    """
    broadcast to connected clients to terminate communication pipes
    """
    shut_down_message = 'SHUTDOWN'
    for client_id in clients_info['clients']:
        clients_info['connections'][client_id].sendall(struct.pack('!I',len(shut_down_message)))
        clients_info['connections'][client_id].sendall(shut_down_message.encode('utf-8'))

def main(server_socket, sub_client):
    #initilization
    clients_info = init_clients_info()
    global_model = MCLR()
    rounds = 100

    #wait for the first client handshake
    server_socket.settimeout(None)
    first_message_time = wait_for_first_handshake(server_socket, clients_info) 

    #30s time window for other clients connections
    while (time.time() - first_message_time < 30):
        try:
            server_socket.settimeout(30 - (time.time() - first_message_time))
            client_socket, _ = server_socket.accept()
            client_thread = threading.Thread(target=handle_client, args=(client_socket, clients_info))
            client_thread.start()
        except socket.timeout:
            pass   

    #for potential client connection attempts after running FL
    stop_event = threading.Event()
    server_socket.settimeout(5)
    new_client_joined = []
    new_client_thread = threading.Thread(target=new_client,args=(server_socket,clients_info,new_client_joined,stop_event))
    new_client_thread.start()

    #FL communication rounds
    print("[SERVER] Broadcasting w0 to all clients registered")
    start_ref = time.time()

    #handles subsampling mode
    sample_sequence = []
    if sub_client:
        if len(clients_info['clients']) < 2:
            sample_list = clients_info['clients']
        else:
            sample_list = random.sample(clients_info['clients'],2)
        sample_sequence.append(sample_list)
    else:
        sample_list = clients_info['clients']

    #broad cast w0 of the global model
    send_global_model(clients_info, global_model,sample_list)

    print("[SERVER] Federated learning is now running")
    for round in range(rounds):
        print("----------------------------")
        print(f"Global Iteration {round + 1}:")
        print(f"Total Number of clients: {len(clients_info['clients'])}")
        
        #receive local models from selected clients
        for client_id in sample_list:
            receive_a_client_model(clients_info,client_id)
        print("Aggregating new global model")

        #aggregate the received local models and update the global model
        global_model = aggregate_models(global_model, clients_info, sample_list)

        #register new clients
        if new_client_joined:
            for conn in new_client_joined:
                handle_client(conn, clients_info)
            new_client_joined = [] 

        #communication round ends
        if round == rounds - 1:
            end_ref = time.time()

            print("Closing all clients")
            terminates_clients(clients_info)

            #prepare average FL evaluation data
            if sample_sequence:
                avg_loss , avg_acc = receive_logs_sub(clients_info, sample_sequence)
            else:
                avg_loss , avg_acc = receive_logs_nonsub(clients_info)
            break

        #broadcast global models to selected clients
        print("Broadcasting new global model")
        if sub_client:
            if len(clients_info['clients']) < 2:
                sample_list = clients_info['clients']
            else:
                sample_list = random.sample(clients_info['clients'],2)
            sample_sequence.append(sample_list)
        send_global_model(clients_info,global_model,sample_list)

    print("[SERVER] All clients have been offline")
    print("[SERVER] Please wait for server roboot. You will get a prompt soon :)")

    #stop accepting new clients
    stop_event.set()
    new_client_thread.join()
    
    return avg_loss, avg_acc, end_ref - start_ref

if __name__ == "__main__":
    """
    help the user to run all 4 Batch GD training
    """
    port = int(sys.argv[1])
    server_socket = prepare_server_socket(port)
    sub_client = int(sys.argv[2])
    if sub_client:
        print("The server will use a subsampling aggregation")
    else:
       print("The server will use a non subsampling aggregation") 
    labels = ["Normal GD", "Mini Batch of 5", "Mini Batch of 10", "Mini Batch of 20"]

    #paint on the canvas
    for i in range(4):
        print(f"[SERVER] Now please run client commands using {labels[i]} flag")
        avg_loss, avg_acc, time_used= main(server_socket, sub_client)
        plt.figure(1,figsize=(11, 5))
        plt.subplot(1,2,1)
        plt.plot(avg_loss, label=f"{labels[i]} ({time_used:.0f}s)", linewidth  = 1)
        plt.legend(loc='upper right', prop={'size': 6}, ncol=2)
        plt.ylabel('Training Loss')
        plt.xlabel('Global rounds')
        plt.subplot(1,2,2)

        plt.plot(avg_acc, label=f"{labels[i]} ({time_used:.0f}s)", linewidth  = 1)
        plt.legend(loc='lower right', prop={'size': 6}, ncol=2)
        plt.ylabel('Training Accuracy')
        plt.xlabel('Global rounds')
    if sub_client:
        plt.suptitle(f"Subsampling of size 2")
    else: 
        plt.suptitle(f"Non subsampling")
    print("Please close the matplotlib GUI to kill this process")
    #show the final result and save it.
    plt.show()
    plt.savefig('comparison.png')
    print("You can still check the digram in 'comparison.png'")
    server_socket.close()
    print(f"Port {port} has been disabled, goodbye!")
