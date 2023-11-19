import socket
import struct
import io
import csv
from PIL import Image
from math import sqrt
import random

# TCP server address and port
SERVER_ADDRESS = 'localhost'
SERVER_PORT = 8080


def receive_frame_data():
    # Create a socket connection
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_ADDRESS, SERVER_PORT))

    print("Connected")
    frames_count = 0

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

            frames_count += 1

            # Write to CSV
            # with open('Temp/Py/frame_data.csv', 'a', newline='') as csvfile:
            #     csv_writer = csv.writer(csvfile)
            #     # csv_writer.writerow([world_id, png_path, reward_signal])
            #     csv_writer.writerow([world_id, None, reward_signal])

            client_socket.sendall(struct.pack('<ii', world_id, random.randint(0, 4)))

    finally:
        client_socket.close()

if __name__ == '__main__':
    receive_frame_data()
