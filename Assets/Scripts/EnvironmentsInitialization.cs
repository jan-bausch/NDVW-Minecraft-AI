using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentsInitialization : MonoBehaviour
{
    public int gridSize = 9;
    public float spacing = 100f;

    void Start()
    {
        GameObject environmentParent = new GameObject("Environments");

        GameObject originalEnvironment = transform.Find("Environment").gameObject;
        originalEnvironment.transform.parent = environmentParent.transform;

        for (int x = 0; x < gridSize; x++)
        {
            for (int z = 0; z < gridSize; z++)
            {
                Vector3 position = new Vector3(x * 20 + x * spacing, 0f, z * 20 + z * spacing);
                GameObject environmentClone = Instantiate(originalEnvironment, position, Quaternion.identity);

                environmentClone.transform.parent = environmentParent.transform;
                environmentClone.SetActive(true);

                int worldSeed = x*10000 + z;
                environmentClone.SendMessage("Generate", worldSeed, SendMessageOptions.RequireReceiver);
            }
        }

        originalEnvironment.SetActive(false);
    }
}
