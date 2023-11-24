import socket
import struct
from PIL import Image
from math import sqrt
import random

# TCP server address and port
SERVER_ADDRESS = 'localhost'
SERVER_PORT = 8080

SEED = 3
BATCH_SIZE = 16
EPISODE_DURATION = 300

def random_model(_worlds_states, _worlds_rewards):
    return [random.randint(0, 4) for _ in range(BATCH_SIZE)]

def neural_model(worlds_states, worlds_rewards):
    pass

def receive_frame_data():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_ADDRESS, SERVER_PORT))

    print("Connected")

    episodes = 0
    frames_per_episode = [0 for _ in range(0, BATCH_SIZE)]
    worlds_data = [[] for _ in range(BATCH_SIZE)]
    client_socket.sendall(struct.pack('<ii', -1*SEED, BATCH_SIZE))

    try:
        while True:
            # Receive data from the server
            data = bytearray()
            while len(data) < 4:  # Read 4 bytes for the data length
                packet = client_socket.recv(4 - len(data))
                print("Packet received")
                if not packet:
                    return
                data.extend(packet)
            length = struct.unpack('<I', data)[0]

            print(f"Message length {length}")

            if length == 0: continue

            # Read the actual data
            data = bytearray()
            while len(data) < length:
                packet = client_socket.recv(length - len(data))
                if not packet:
                    print("Packet was not as large as planned")
                    print(data)
                    return
                data.extend(packet)

            # Unpack received binary data
            world_id, reward_signal, *pixels_grayscale = struct.unpack(f'<if{len(data)//4 - 2}i', data)
            print(world_id, reward_signal)

            # # Convert pixels to PNG
            # image = Image.new('L', (int(sqrt(len(pixels_grayscale))), int(sqrt(len(pixels_grayscale)))))
            # #print(pixels_grayscale)
            # image.putdata(pixels_grayscale)
            # png_path = f'Temp/Py/frame_{frames_count}_world_{world_id}.png'
            # image.save(png_path)

            worlds_data

            frames_per_episode[world_id] += 1

            if frames_per_episode[world_id] > EPISODE_DURATION:
                episodes += 1
                frames_per_episode = [0 for _ in range(0, BATCH_SIZE)]
                worlds_data = [[] for _ in range(BATCH_SIZE)]
                client_socket.sendall(struct.pack('<ii', -1*(SEED+episodes), BATCH_SIZE))
                continue

            worlds_data[world_id].append({
                "state": pixels_grayscale,
                "reward": reward_signal
            })

            batch_ready = True
            for world_id in range(BATCH_SIZE):
                if len(worlds_data[world_id]) == 0:
                    batch_ready = False
                    break
            
            if batch_ready:
                worlds_states = [None for _ in range(BATCH_SIZE)]
                worlds_rewards = [None for _ in range(BATCH_SIZE)]
                for world_id in range(BATCH_SIZE):
                    worlds_states[world_id] = worlds_data[world_id][0]["state"]
                    worlds_rewards[world_id] = worlds_data[world_id][0]["reward"]
                    worlds_data[world_id].pop(0)

                actions = random_model(worlds_states, worlds_rewards)
                for world_id in range(BATCH_SIZE):      
                    client_socket.sendall(struct.pack('<ii', world_id, actions[world_id]))

    finally:
        client_socket.close()

if __name__ == '__main__':
    receive_frame_data()
