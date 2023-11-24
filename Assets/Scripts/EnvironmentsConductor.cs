using System.Collections;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System;
using UnityEngine;

namespace Environment
{
    public class EnvironmentsConductor : MonoBehaviour
    {
        public int maxFramesAccumulation = 1;

        public bool allAtTheSameTime = false;

        public bool queueControlled = true;

        private int frameCount = 0;

        private ConcurrentDictionary<int, ConcurrentQueue<EnvironmentData>> environmentQueues = new ConcurrentDictionary<int, ConcurrentQueue<EnvironmentData>>();

        private ConcurrentDictionary<int, ConcurrentQueue<int>> playerActionQueues = new ConcurrentDictionary<int, ConcurrentQueue<int>>();
        private ConcurrentQueue<(int, int)> targetedRequestsQueue = new ConcurrentQueue<(int, int)>();


        void Start()
        {
            if (!queueControlled)
            {
                ResetAndGenerate(4, 0);
            }
        }

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

        private void ResetAndGenerate(int worldsCount, int seed)
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

            var initializer = gameObject.GetComponent<EnvironmentsInitialization>();
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
                if ((queueControlled || environmentQueues[id].Count < maxFramesAccumulation)
                    && (!queueControlled || playerActionQueues[id].TryDequeue(out action)))
                {
                    EnvironmentCamera cam = env.gameObject.GetComponent<EnvironmentCamera>();
                    int[] pixelsGrayscale = cam.GetPixelsGrayscale();
                    //Debug.Log("Frame:" + frameCount + " Child:" + id + "(" + x + ", " + z + ")" + " Pixels count: " + pixelsGrayscale.Length);

                    Transform playerTransform = env.Find("Player");
                    EnvironmentPlayer player = playerTransform.gameObject.GetComponent<EnvironmentPlayer>();

                    if (queueControlled && action != -1)
                    {
                        player.goingBackward = false;
                        player.goingForward = false;
                        player.goingLeft = false;
                        player.goingRight = false;
                        player.jumping = false;

                        if (action == 0) player.goingBackward = true;
                        if (action == 1) player.goingForward = true;
                        if (action == 2) player.goingLeft = true;
                        if (action == 3) player.goingRight = true;
                        if (action == 4) player.jumping = true;

                        player.MoveUpdate();
                        // later: update enemy here
                    } else if (!queueControlled) {
                        player.MoveUpdate();
                        // later: update enemy here
                    }

                    EnvironmentReward rewardEmitter = env.gameObject.GetComponent<EnvironmentReward>();
                    float rewardSignal = rewardEmitter.GetReward();
                    //Debug.Log("Frame:" + frameCount + " Child:" + id + "(" + x + ", " + z + ")" + "Reward: " + rewardSignal);

                    EnvironmentData frameInfo = new EnvironmentData
                    {
                        pixelsGrayscale = pixelsGrayscale,
                        rewardSignal = rewardSignal
                    };

                    environmentQueues[id].Enqueue(frameInfo);
                }
            }
        }


    }
}
