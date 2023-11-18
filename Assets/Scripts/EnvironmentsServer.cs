using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Environment;
using System;
using System.Linq;

public class EnvironmentsServer : MonoBehaviour
{
    private int frameCount = 0;

    void Update()
    {
        frameCount++;

        // Find the "Environments" GameObject in the scene
        GameObject environments = GameObject.Find("Environments");

        // Ensure "Environments" is found before proceeding
        if (environments != null)
        {
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
                Debug.Log(gridSize);
                if (frameCount % children.Length != id) continue;
                EnvironmentCamera cam = env.gameObject.GetComponent<EnvironmentCamera>();
                int[] pixelsGrayscale = cam.GetPixelsGrayscale();
                Debug.Log("Frame:" + frameCount + " Child:" + id + "(" + x + ", " + z + ")" + " Pixels count: " + pixelsGrayscale.Length);
            
                Transform playerTransform = env.Find("Player");
                EnvironmentPlayer player = playerTransform.gameObject.GetComponent<EnvironmentPlayer>();
                player.MoveUpdate();

                EnvironmentReward rewardEmitter = env.gameObject.GetComponent<EnvironmentReward>();
                float rewardSignal = rewardEmitter.GetReward();
                Debug.Log("Frame:" + frameCount + " Child:" + id + "(" + x + ", " + z + ")" + "Reward: " + rewardSignal);
            }
        }
    }
}
