using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Environment;
using WorldEditing;

public class BlockInteraction : MonoBehaviour
{
    [Header("Camera")]
    public Camera playerCamera;
    public float rayTime = 1.0f;
    public float range = 5.0f;

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0)) // Assuming left mouse button for breaking
        {
            Ray ray = playerCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            Debug.DrawRay(ray.origin, ray.direction * 100f, Color.red, rayTime);

            if (Physics.Raycast(ray, out hit, range))
            {
                Vector3 hitPointOnMesh = hit.point;

                // Round the coordinates to integers
                int roundedX = Mathf.FloorToInt(hitPointOnMesh.x);
                int roundedY = Mathf.FloorToInt(hitPointOnMesh.y);
                int roundedZ = Mathf.FloorToInt(hitPointOnMesh.z);

                Vector3 roundedHitPoint = new Vector3(roundedX, roundedY, roundedZ);

                Debug.Log(roundedHitPoint);

                // editableWorld.SetBlock(roundedX, roundedY, roundedZ, -1);
                // EnvironmentWorldGeneration environmentWorldGeneration = GetComponent<EnvironmentWorldGeneration>();
                EnvironmentWorldGeneration environmentWorldGeneration = FindObjectOfType<EnvironmentWorldGeneration>();
                // Debug.Log(environmentWorldGeneration);
                EditableWorld world = environmentWorldGeneration.world;
                // Debug.Log(world);
                world.SetBlock(roundedX, roundedY, roundedZ, -1);
                Debug.Log(gameObject);

                // VoxelRenderer.RenderWorld(gameObject, world, material);
            }
        }
    }

    // private void placeCursorBlocks(){
    //     float step = checkIncrement;
    //     Vector3 lastPos = new Vector3;

    //     while (step < reach) {
    //         Vector3 pos = playerCamera.position * (playerCamera.forward * step)

    //     }
    // }
}