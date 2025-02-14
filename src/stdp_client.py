#!/usr/bin/env python3
import socket
import sys
import threading
import json
import time
import numpy as np
import torch
from stdp_model import SNNControllerSTDP

# Set device (CPU)
device = torch.device("cpu")

# Load the SNN controller model (STDP enabled).
# In a training scenario, you might load pre-trained weights or start from random.
model = SNNControllerSTDP(input_dim=24, hidden_dim=32, output_dim=4).to(device)
model.train()  # In STDP, learning occurs during forward passes.
print("[Client] SNN Controller with STDP loaded.")

# Constants for central server connection.
SERVER_HOST = "192.168.1.100"  # Change to the central server's IP.
SERVER_PORT = 9999

# Base port for peer communications.
PEER_PORT_BASE = 10000

peer_messages = []  # Shared list for messages from peers.

#############################################
# Peer Server
#############################################
def peer_server(env_id):
    listen_port = PEER_PORT_BASE + env_id
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("", listen_port))
    server_sock.listen(5)
    print(f"[PeerServer] Env {env_id} listening on port {listen_port}")
    while True:
        try:
            conn, addr = server_sock.accept()
            print(f"[PeerServer] Received peer connection from {addr}")
            threading.Thread(target=handle_peer_connection, args=(conn,), daemon=True).start()
        except Exception as e:
            print(f"[PeerServer] Exception: {e}")
            break

def handle_peer_connection(conn):
    try:
        while True:
            data = ""
            while "\n" not in data:
                chunk = conn.recv(4096).decode()
                if not chunk:
                    raise ConnectionError("Peer disconnected")
                data += chunk
            msg = json.loads(data.strip())
            print(f"[PeerServer] Received peer message: {msg}")
            peer_messages.append(msg)
    except Exception as e:
        print(f"[PeerServer] Peer connection closed: {e}")
    finally:
        conn.close()

#############################################
# Peer Client
#############################################
def connect_to_peers(peer_list):
    peer_conns = {}
    for peer_str in peer_list:
        try:
            ip, peer_id_str = peer_str.split(":")
            peer_id = int(peer_id_str)
            remote_port = PEER_PORT_BASE + peer_id
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, remote_port))
            peer_conns[peer_str] = sock
            print(f"[PeerClient] Connected to peer {peer_str} on port {remote_port}")
        except Exception as e:
            print(f"[PeerClient] Could not connect to peer {peer_str}: {e}")
    return peer_conns

def broadcast_to_peers(peer_conns, message):
    data = (json.dumps(message) + "\n").encode()
    for peer_str, sock in peer_conns.items():
        try:
            sock.sendall(data)
        except Exception as e:
            print(f"[PeerClient] Error sending to {peer_str}: {e}")

#############################################
# Main Client: Connect to central server and run SNN inference with STDP
#############################################
def run_client(env_id, peer_list):
    threading.Thread(target=peer_server, args=(env_id,), daemon=True).start()
    time.sleep(1)
    peer_conns = connect_to_peers(peer_list)
    
    # Connect to central server.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_HOST, SERVER_PORT))
    print(f"[Client {env_id}] Connected to central server at {SERVER_HOST}:{SERVER_PORT}")
    s.sendall(f"{env_id}\n".encode())
    
    try:
        while True:
            # Receive observation from the central server.
            data = ""
            while "\n" not in data:
                chunk = s.recv(4096).decode()
                if not chunk:
                    raise ConnectionError("Central server disconnected")
                data += chunk
            obs_msg = json.loads(data.strip())
            obs = obs_msg.get("observation", None)
            print(f"[Client {env_id}] Received observation: {obs}")
            
            # Broadcast observation to peers.
            peer_msg = {"env_id": env_id, "observation": obs}
            broadcast_to_peers(peer_conns, peer_msg)
            
            if peer_messages:
                print(f"[Client {env_id}] Peer messages: {peer_messages}")
                peer_messages.clear()
            
            # Convert observation to a torch tensor.
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # shape: (1,24)
            with torch.no_grad():
                # Forward pass through SNN controller with STDP.
                action_tensor = model(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy().tolist()
            print(f"[Client {env_id}] Computed action: {action}")
            s.sendall((json.dumps(action) + "\n").encode())
            
            # Receive step result from server.
            data = ""
            while "\n" not in data:
                chunk = s.recv(4096).decode()
                if not chunk:
                    raise ConnectionError("Central server disconnected")
                data += chunk
            step_msg = json.loads(data.strip())
            print(f"[Client {env_id}] Step result: {step_msg}")
    except Exception as e:
        print(f"[Client {env_id}] Exception: {e}")
    finally:
        print(f"[Client {env_id}] Closing connection to central server.")
        s.close()
        for sock in peer_conns.values():
            sock.close()

if __name__ == "__main__":
    # Usage: python env_client_peer_snn.py <env_id> <peer1,peer2,...>
    if len(sys.argv) < 3:
        print("Usage: python env_client_peer_snn.py <env_id> <peer1,peer2,...>")
        sys.exit(1)
    env_id = int(sys.argv[1])
    peer_list = sys.argv[2].split(",")
    run_client(env_id, peer_list)
