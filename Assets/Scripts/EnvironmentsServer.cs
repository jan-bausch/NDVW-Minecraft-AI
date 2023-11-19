using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Linq;
using UnityEngine;

namespace Environment
{
    public class EnvironmentsServer : MonoBehaviour
    {
        private TcpListener server;
        private CancellationTokenSource cancellationTokenSource;

        private List<TcpClient> connectedClients = new List<TcpClient>();

        private async void Start()
        {
            cancellationTokenSource = new CancellationTokenSource();
            // Start TCP server
            await StartServerAsync();

            // Start handling sending updates to clients in a separate task
            _ = Task.Run(async () => await SendUpdatesAsync(), cancellationTokenSource.Token);
        }

        private async Task StartServerAsync()
        {
            try
            {
                server = new TcpListener(IPAddress.Any, 8080);
                server.Start();
                Debug.Log("Server started. Waiting for connections...");

                while (true)
                {
                    TcpClient client = await server.AcceptTcpClientAsync();
                    Debug.Log("Client connected: " + client.Client.RemoteEndPoint);

                    connectedClients.Add(client);

                    // Handle each client connection in a separate task
                    _ = Task.Run(() => HandleClientAsync(client), cancellationTokenSource.Token);
                }
            }
            catch (Exception e)
            {
                Debug.LogError("Server error: " + e.Message);
            }
        }

        private async Task HandleClientAsync(TcpClient client)
        {
            try
            {
                byte[] buffer = new byte[1024];
                int bytesRead;
                NetworkStream stream = client.GetStream();

                while ((bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length, cancellationTokenSource.Token)) > 0)
                {
                    string receivedMessage = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    Debug.Log("Received message from client: " + receivedMessage);
                }
            }
            catch (Exception e)
            {
                Debug.LogError("Client error: " + e.Message);
            }
            finally
            {
                connectedClients.Remove(client);
                Debug.Log("Client disconnected: " + client.Client.RemoteEndPoint);
                client.Close();
            }
        }

        private async Task SendUpdatesAsync()
        {
            var cancellationToken = cancellationTokenSource.Token;
            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    EnvironmentsConductor conductor = gameObject.GetComponent<EnvironmentsConductor>();
                    var environmentQueues = conductor.GetEnvironmentQueues();
                    foreach (var queue in environmentQueues)
                    {
                        if (queue.Value.TryDequeue(out EnvironmentData frameInfo))
                        {
                            EnvironmentDataSerializer serializer = new EnvironmentDataSerializer();

                            int imageId = 1;
                            float rewardSignal = frameInfo.rewardSignal;
                            int[] pixelsGrayscale = frameInfo.pixelsGrayscale;

                            // Serialize frameInfo to binary
                            byte[] serializedData = serializer.SerializeFrameInfo(imageId, rewardSignal, pixelsGrayscale);

                            foreach (TcpClient client in connectedClients)
                            {
                                await client.GetStream().WriteAsync(serializedData, 0, serializedData.Length, cancellationToken);
                            }
                        }
                    }

                    // await Task.Delay(1000/60, cancellationToken);
                }
            }
            catch (Exception e)
            {
                Debug.LogError("Error sending updates: " + e.Message);
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