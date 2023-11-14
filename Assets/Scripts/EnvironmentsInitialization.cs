using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentsInitialization : MonoBehaviour
{
    // Start is called before the first frame update    
    public int gridSize = 9;
    public float spacing = 100f;

    void Start()
    {
        // Spawn the initial Environment
        GameObject environmentParent = new GameObject("Environments");

        // Get the original "Environment" GameObject
        GameObject originalEnvironment = transform.Find("Environment").gameObject;
        originalEnvironment.transform.parent = environmentParent.transform;

        // Loop to spawn clones in a grid
        for (int x = 0; x < gridSize; x++)
        {
            for (int z = 0; z < gridSize; z++)
            {
                // Calculate the position for each clone
                Vector3 position = new Vector3(x * 20 + x * spacing, 0f, z * 20 + z * spacing);

                // Instantiate a clone of the original Environment GameObject
                GameObject environmentClone = Instantiate(originalEnvironment, position, Quaternion.identity);

                // Set the clone as a child of the parent GameObject
                environmentClone.transform.parent = environmentParent.transform;
                environmentClone.SetActive(true);

                environmentClone.SendMessage("Generate", SendMessageOptions.RequireReceiver);
            }
        }

        originalEnvironment.SetActive(false);
    }
}
