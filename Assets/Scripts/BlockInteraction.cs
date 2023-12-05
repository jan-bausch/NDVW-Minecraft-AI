using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Environment;
using WorldEditing;
using Voxels;

public class BlockInteraction : MonoBehaviour
{
    [Header("Camera")]
    public Camera playerCamera;

    [Header("Block Placement")]
    public float rayTime = 1.0f;
    public float range = 5.0f;
    public float checkIncrement = 0.1f;

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0)){
            removeBlock();
        } else if (Input.GetMouseButtonDown(1)){
            placeBlock();
        }
        // if (Input.GetMouseButtonDown(0)) // Assuming left mouse button for breaking
        // {
        //     Ray ray = playerCamera.ScreenPointToRay(Input.mousePosition);
        //     RaycastHit hit;

        //     Debug.DrawRay(ray.origin, ray.direction * 100f, Color.red, rayTime);

        //     if (Physics.Raycast(ray, out hit, range))
        //     {
        //         Vector3 hitPointOnMesh = hit.point;

        //         // Round the coordinates to integers
        //         int roundedX = Mathf.FloorToInt(hitPointOnMesh.x);
        //         int roundedY = Mathf.FloorToInt(hitPointOnMesh.y);
        //         int roundedZ = Mathf.FloorToInt(hitPointOnMesh.z);

        //         Vector3 roundedHitPoint = new Vector3(roundedX, roundedY, roundedZ);

        //         Debug.Log(roundedHitPoint);

        //         EnvironmentWorldGeneration environmentWorldGeneration = FindObjectOfType<EnvironmentWorldGeneration>();
        //         EditableWorld world = environmentWorldGeneration.world;
        //         // Debug.Log("Before " + world.BlockAt(roundedX, roundedY, roundedZ));
        //         world.SetBlock(roundedX, roundedY, roundedZ, -1);
        //         // Debug.Log("After" + world.BlockAt(roundedX, roundedY, roundedZ));
        //         Debug.Log(environmentWorldGeneration.getGameObject());
        //         VoxelRenderer.UpdateWorld(environmentWorldGeneration.getGameObject(), world, environmentWorldGeneration.material);
        //     }
        // }
    }

    private void placeBlock()
    {
        var result = executeRay();
        if (result != null){
            (Vector3 pos, Vector3 lastPos) = result.Value;
            Debug.Log(pos + " " + lastPos);

            int roundedX = Mathf.FloorToInt(lastPos.x);
            int roundedY = Mathf.FloorToInt(lastPos.y);
            int roundedZ = Mathf.FloorToInt(lastPos.z);

            // only one world
            EnvironmentWorldGeneration environmentWorldGeneration = FindObjectOfType<EnvironmentWorldGeneration>();
            EditableWorld world = environmentWorldGeneration.world;
            world.SetBlock(roundedX, roundedY, roundedZ, VoxelWorld.SOLID_PRECIOUS);
        }
    }

    private void removeBlock(){
        var result = executeRay();
        if (result != null){
            (Vector3 pos, Vector3 lastPos) = result.Value;
            Debug.Log(pos + " " + lastPos);

            
            int roundedX = Mathf.FloorToInt(pos.x);
            int roundedY = Mathf.FloorToInt(pos.y);
            int roundedZ = Mathf.FloorToInt(pos.z);
            
            // only one world
            EnvironmentWorldGeneration environmentWorldGeneration = FindObjectOfType<EnvironmentWorldGeneration>();
            EditableWorld world = environmentWorldGeneration.world;
            world.SetBlock(roundedX, roundedY, roundedZ, VoxelWorld.AIR);


            // int roundedX = Mathf.FloorToInt(pos.x);
            // int roundedY = Mathf.FloorToInt(pos.y);
            // int roundedZ = Mathf.FloorToInt(pos.z);
            // EnvironmentWorldGeneration environmentWorldGeneration = FindObjectOfType<EnvironmentWorldGeneration>();
            // EditableWorld world = environmentWorldGeneration.world;
            
            // Debug.Log(world.BlockAt(roundedX, roundedY, roundedZ));
            // world.SetBlock(roundedX, roundedY, roundedZ, 1);
            // Debug.Log(world.BlockAt(roundedX, roundedY, roundedZ));
            // environmentWorldGeneration.UpdateEnvironment();
            // Debug.Log(world.BlockAt(roundedX, roundedY, roundedZ));

            // VoxelRenderer.UpdateWorld(environmentWorldGeneration.getGameObject(), world, environmentWorldGeneration.material);  
            // GameObject voxel = VoxelRenderer.renderVoxel(world, roundedX, roundedY, roundedZ);
            // if (voxel != null) 
            // {
            //     voxel.transform.SetParent(environmentWorldGeneration.getGameObject().transform);
            // }
            // VoxelRenderer.RenderWorld(environmentWorldGeneration.getGameObject(), world, environmentWorldGeneration.material); 
            // Debug.Log(world.BlockAt(roundedX, roundedY, roundedZ));
        }
    }

    private (Vector3, Vector3)? executeRay(){
        float step = checkIncrement;
        Vector3 lastPos = new Vector3();
        EnvironmentWorldGeneration environmentWorldGeneration = FindObjectOfType<EnvironmentWorldGeneration>();
        EditableWorld world = environmentWorldGeneration.world;

        while (step < range) {
            Vector3 pos = playerCamera.transform.position + (playerCamera.transform.forward * step);
            if (world.BlockAt(Mathf.FloorToInt(pos.x), Mathf.FloorToInt(pos.y), Mathf.FloorToInt(pos.z)) != -1) {
                Debug.DrawRay(playerCamera.transform.position, playerCamera.transform.forward * step * 100f, Color.red, rayTime);
                return (pos, lastPos);
            }
            lastPos = new Vector3(Mathf.FloorToInt(pos.x), Mathf.FloorToInt(pos.y), Mathf.FloorToInt(pos.z));
            step += checkIncrement;
        }
        return null;
    }
}
