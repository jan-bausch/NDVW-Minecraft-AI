using System.Collections;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System;
using UnityEngine;

namespace Environment {
public class EnvironmentsConductor : MonoBehaviour
{
    public int maxFramesAccumulation = 1;

    public bool allAtTheSameTime = false;

    private int frameCount = 0;

    private Dictionary<int, ConcurrentQueue<EnvironmentData>> environmentQueues = new Dictionary<int, ConcurrentQueue<EnvironmentData>>();

    public Dictionary<int, ConcurrentQueue<EnvironmentData>> GetEnvironmentQueues()
    {
        return environmentQueues;
    }

    void Update()
    {
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

            if (!environmentQueues.ContainsKey(id))
            {
                environmentQueues[id] = new ConcurrentQueue<EnvironmentData>();
            }

            if (!allAtTheSameTime && frameCount % children.Length != id) continue;

            if (environmentQueues[id].Count < maxFramesAccumulation)
            {
                EnvironmentCamera cam = env.gameObject.GetComponent<EnvironmentCamera>();
                int[] pixelsGrayscale = cam.GetPixelsGrayscale();
                //Debug.Log("Frame:" + frameCount + " Child:" + id + "(" + x + ", " + z + ")" + " Pixels count: " + pixelsGrayscale.Length);

                Transform playerTransform = env.Find("Player");
                EnvironmentPlayer player = playerTransform.gameObject.GetComponent<EnvironmentPlayer>();
                player.MoveUpdate();

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
