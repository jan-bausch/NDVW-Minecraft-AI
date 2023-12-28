using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Linq;
using Environment;
using UnityEngine;

namespace Multienv
{
    public class MultienvServer : MonoBehaviour
    {
        private TcpListener server;
        private CancellationTokenSource cancellationTokenSource;

        private async void Start()
        {
            cancellationTokenSource = new CancellationTokenSource();
            // Start TCP server
            await StartServerAsync();            
        }

        private async Task StartServerAsync()
        {
            int startingPort = 8080;
            int maxAttempts = 100;
            bool portAvailable = false;

            for (int port = startingPort; port < startingPort + maxAttempts; port++)
            {
                server = new TcpListener(IPAddress.Parse("0.0.0.0"), port);

                try
                {
                    server.Start();
                    portAvailable = true;
                    break;
                }
                catch
                {
                    server.Stop();
                }
            }

            if (!portAvailable)
            {
                Console.WriteLine("Could not start the server on any available port.");
                return;
            }
            
            try
            {                
                Debug.Log("Server is running on:");
                Debug.Log("Address: " + server.LocalEndpoint);
                Debug.Log("Port: " + ((IPEndPoint)server.LocalEndpoint).Port);

                while (true)
                {
                    TcpClient client = await server.AcceptTcpClientAsync();
                    Debug.Log("Client connected: " + client.Client.RemoteEndPoint);

                    MultienvConductor conductor = gameObject.GetComponent<MultienvConductor>();

                    // Handle each client connection in a separate task
                    _ = Task.Run(() => HandleClientAsync(client, conductor), cancellationTokenSource.Token);
                    _ = Task.Run(() => SendUpdatesAsync(client, conductor), cancellationTokenSource.Token);
                }
            }
            catch (Exception e)
            {
                Debug.LogError("Server error: " + e.Message);
            }
        }

        private async Task HandleClientAsync(TcpClient client, MultienvConductor conductor)
        {
            try
            {
                byte[] buffer = new byte[8];
                int bytesRead;
                NetworkStream stream = client.GetStream();

                while ((bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length, cancellationTokenSource.Token)) > 0)
                {
                    byte[] firstHalf = new byte[buffer.Length / 2];
                    byte[] secondHalf = new byte[buffer.Length - firstHalf.Length];

                    Array.Copy(buffer, firstHalf, firstHalf.Length);
                    Array.Copy(buffer, firstHalf.Length, secondHalf, 0, secondHalf.Length);
                    
                    int target = BitConverter.ToInt32(firstHalf, 0);
                    int request = BitConverter.ToInt32(secondHalf, 0);
                    //Debug.Log("Received message from client: " + worldId + ", " + action);
                    
                    conductor.HandleIncomingRequest(target, request);
                }
            }
            catch (Exception e)
            {
                Debug.LogError("Client error: " + e.Message);
            }
            finally
            {
                Debug.Log("Client disconnected: " + client.Client.RemoteEndPoint);
                client.Close();
            }
        }

        private async Task SendUpdatesAsync(TcpClient client, MultienvConductor conductor)
        {
            var cancellationToken = cancellationTokenSource.Token;
            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    var environmentQueues = conductor.GetEnvironmentQueues();
                    foreach (var worldId in environmentQueues.Keys.ToList())
                    {
                        //Debug.Log(worldId);
                        var queue = environmentQueues[worldId];
                        if (queue.TryDequeue(out EnvironmentData frameInfo))
                        {
                            EnvironmentDataSerializer serializer = new EnvironmentDataSerializer();

                            float rewardSignal = frameInfo.rewardSignal;
                            int[] gameState = frameInfo.gameState;

                            // Serialize frameInfo to binary
                            byte[] serializedData = serializer.SerializeFrameInfo(worldId, rewardSignal, gameState);

                            byte[] sendData = new byte[sizeof(int) + serializedData.Length];
                            Buffer.BlockCopy(BitConverter.GetBytes(serializedData.Length), 0, sendData, 0, sizeof(int));
                            Buffer.BlockCopy(serializedData, 0, sendData, sizeof(int), serializedData.Length);

                            await client.GetStream().WriteAsync(sendData, 0, sendData.Length, cancellationToken);
                            //Debug.Log("Sent " + serializedData.Length);
                        }
                    }

                    // await Task.Delay(1000/60, cancellationToken);
                }
            }
            catch (Exception e)
            {
                Debug.LogError("Error sending updates: " + e.Message + e.StackTrace);
                
            }
        }

        private void OnDestroy()
        {
            // Cleanup resources when the script is destroyed
            cancellationTokenSource?.Cancel();
            server?.Stop();
        }
    }
}