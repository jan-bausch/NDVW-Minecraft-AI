using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;

namespace Environment
{

public class EnvironmentsInitialization : MonoBehaviour
{
    public float spacing = 100f;

    public void Generate(int worldsCount, int seed)
    {
        int gridSize = (int) Math.Sqrt(worldsCount);

        GameObject environmentParent = GameObject.Find("Environments");
        if (environmentParent is null)
        {
            environmentParent = new GameObject("Environments");
        } else 
        {
            foreach (Transform child in environmentParent.transform) {
                GameObject.Destroy(child.gameObject);
            }
        }

        GameObject originalEnvironment = transform.Find("Environment").gameObject;

        Debug.Log(worldsCount + ", " + seed);

        for (int x = 0; x < gridSize; x++)
        {
            for (int z = 0; z < gridSize; z++)
            {
                Vector3 position = new Vector3(x * 20 + x * spacing, 0f, z * 20 + z * spacing);
                GameObject environmentClone = Instantiate(originalEnvironment, position, Quaternion.identity);

                environmentClone.transform.parent = environmentParent.transform;
                environmentClone.SetActive(true);
                environmentClone.name = "Environment_" + x + "_" + z; 

                int worldSeed = seed*(1000*1000)+x*(1000)+z;
                environmentClone.SendMessage("Generate", worldSeed, SendMessageOptions.RequireReceiver);
            }
        }

        originalEnvironment.SetActive(false);
    }
}
}