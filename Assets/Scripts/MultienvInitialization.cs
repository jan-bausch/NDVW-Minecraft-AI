using System.Collections;
using System.Collections.Generic;
using System;
using Environment;
using UnityEngine;

namespace Multienv
{
    public class MultienvInitialization : MonoBehaviour
    {
        public int gridSize = 1;
        public int seed = 0;
        public float spacing = 100f;

        public bool remoteControlled = false; 

        void Start()
        {
            if (!remoteControlled)
            {
                Generate(gridSize*gridSize, 0);
            }
            else
            {
                MultienvConductor cond = gameObject.AddComponent<MultienvConductor>();
                MultienvServer server = gameObject.AddComponent<MultienvServer>();
            }
        }

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
            EnvironmentWorldGeneration envworld = originalEnvironment.GetComponent<EnvironmentWorldGeneration>();

            Debug.Log(worldsCount + ", " + seed);

            for (int x = 0; x < gridSize; x++)
            {
                for (int z = 0; z < gridSize; z++)
                {
                    
                    Vector3 position = new Vector3(x * envworld.worldSizeX + x * spacing, 0f, z * envworld.worldSizeZ + z * spacing);
                    GameObject environmentClone = Instantiate(originalEnvironment, position, Quaternion.identity);

                    environmentClone.transform.parent = environmentParent.transform;

                    if ((x != 0 || z != 0) && !remoteControlled)
                    {
                        Transform playerTransform = environmentClone.transform.Find("Player");
                        playerTransform.gameObject.SetActive(false);
                        Transform camTransform = environmentClone.transform.Find("CameraHolder").Find("Camera");
                        camTransform.gameObject.SetActive(false);
                    }


                    environmentClone.SetActive(true);
                    environmentClone.name = "Environment_" + x + "_" + z; 

                    int worldSeed = seed*(1000*1000)+x*(1000)+z;
                    environmentClone.SendMessage("Generate", worldSeed, SendMessageOptions.RequireReceiver);
                    
                    if (remoteControlled)
                    {
                        if (x != 0 || z != 0)
                        {
                            Transform camTransform = environmentClone.transform.Find("CameraHolder").Find("Camera");
                            AudioListener al = camTransform.gameObject.GetComponent<AudioListener>();
                            Destroy(al);
                        }

                        EnvironmentIO envio = environmentClone.GetComponent<EnvironmentIO>();
                        envio.EnableRemoteControlled();
                    }
                }
            }

            originalEnvironment.SetActive(false);
        }
    }
}
