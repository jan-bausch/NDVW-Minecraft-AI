using System.Collections;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System;
using UnityEngine;
using Environment;

namespace Multienv
{
    public class MultienvConductor : MonoBehaviour
    {
        public int maxFramesAccumulation = 1;

        public bool allAtTheSameTime = false;

        private int frameCount = 0;

        private ConcurrentDictionary<int, ConcurrentQueue<EnvironmentData>> environmentQueues = new ConcurrentDictionary<int, ConcurrentQueue<EnvironmentData>>();

        private ConcurrentDictionary<int, ConcurrentQueue<int>> playerActionQueues = new ConcurrentDictionary<int, ConcurrentQueue<int>>();
        private ConcurrentQueue<(int, int)> targetedRequestsQueue = new ConcurrentQueue<(int, int)>();

        public ConcurrentDictionary<int, ConcurrentQueue<EnvironmentData>> GetEnvironmentQueues()
        {
            return environmentQueues;
        }

        public void HandleIncomingRequest(int target, int request)
        {
            if (target >= 0)
            {
                int worldId = target;
                int action = request;
                playerActionQueues[worldId].Enqueue(action);
            } else 
            {
                targetedRequestsQueue.Enqueue((target*-1, request));
            }
        }

        public void ResetAndGenerate(int worldsCount, int seed)
        {
            for (int i = 0; i < worldsCount; i++)
            {
                environmentQueues[i] = new ConcurrentQueue<EnvironmentData>();
                playerActionQueues[i] = new ConcurrentQueue<int>();
                playerActionQueues[i].Enqueue(-1);
            }
            // environmentQueues = new Dictionary<int, ConcurrentQueue<EnvironmentData>>();
            // playerActionQueues = new Dictionary<int, ConcurrentQueue<int>>();
            targetedRequestsQueue = new ConcurrentQueue<(int, int)>();

            frameCount = 0;

            var initializer = gameObject.GetComponent<MultienvInitialization>();
            initializer.Generate(worldsCount, seed);
        }

        void Update()
        {
            var targetedRequest = (-1, -1);
            if (targetedRequestsQueue.TryDequeue(out targetedRequest))
            {
                // for now only regenerate requests exist
                var (target, request) = targetedRequest;
                int seed = target;
                int worldsCount = request;

                ResetAndGenerate(worldsCount, seed);
                return;
            }

            frameCount++;

            // Find the "Environments" GameObject in the scene
            GameObject environments = GameObject.Find("Environments");

            // Ensure "Environments" is found before proceeding
            if (environments == null) return;

            Transform[] children = environments.GetComponentsInChildren<Transform>()
                .Where(child => child != environments.transform && child.name.StartsWith("Environment_"))
                .ToArray();

            foreach (Transform env in children)
            {
                string[] nameParts = env.name.Split('_'); // Split the name by underscore
                int x = 0, z = 0;
                int.TryParse(nameParts[1], out x);
                int.TryParse(nameParts[2], out z);
                int gridSize = (int)Math.Sqrt(children.Length);
                int id = x * gridSize + z;
                //Debug.Log(gridSize);

                if (!environmentQueues.ContainsKey(id) || !playerActionQueues.ContainsKey(id))
                {
                    environmentQueues[id] = new ConcurrentQueue<EnvironmentData>();
                    playerActionQueues[id] = new ConcurrentQueue<int>();
                    playerActionQueues[id].Enqueue(-1);
                }

                if (!allAtTheSameTime && frameCount % children.Length != id) continue;
                
                int action = -1;
                if (environmentQueues[id].Count < maxFramesAccumulation)
                {
                    EnvironmentIO envio = env.gameObject.GetComponent<EnvironmentIO>();
                    int[] gameState = envio.GetGameState();

                    if (action != -1)
                    {
                        envio.ResetAllInputs();
                        envio.ToggleInput(action);
                        envio.MoveUpdate(0.1f);                    
                    }

                    float rewardSignal = envio.GetReward();

                    EnvironmentData frameInfo = new EnvironmentData
                    {
                        gameState = gameState,
                        rewardSignal = rewardSignal
                    };

                    environmentQueues[id].Enqueue(frameInfo);
                }
            }
        }


    }
}