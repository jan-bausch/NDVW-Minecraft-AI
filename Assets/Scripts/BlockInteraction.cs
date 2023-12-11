using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Environment;
using WorldEditing;
using Voxels;

namespace Player
{
    public class BlockInteraction : MonoBehaviour
    {
        [Header("Camera")]
        public Camera playerCamera;

        [Header("Block Placement")]
        public bool remoteControlled = false;
        public bool placingBlock = false;
        public bool breakingBlock = false;
        public float rayTime = 1.0f;
        public float range = 5.0f;
        public float checkIncrement = 0.1f;

        private int invSolid = 0;
        private int invPrecious = 0;

        void Update()
        {
            if (!remoteControlled)
            {
                if (Input.GetMouseButtonDown(0)){
                    removeBlock();
                } else if (Input.GetMouseButtonDown(1)){
                    placeBlock();
                }
            }
        }

        public void MoveUpdate(float delta)
        {
            if (placingBlock)
            {
                placeBlock();
            }
            if (breakingBlock)
            {
                removeBlock();
            }
        }

        public (int, int) GetInv() {
            return (invSolid, invPrecious);
        }

        private void placeBlock()
        {
            var result = executeRay();
            if (result != null){
                (Vector3 pos, Vector3 lastPos) = result.Value;

                int roundedX = Mathf.FloorToInt(lastPos.x);
                int roundedY = Mathf.FloorToInt(lastPos.y);
                int roundedZ = Mathf.FloorToInt(lastPos.z);

                // only one world
                EnvironmentWorldGeneration environmentWorldGeneration = transform.parent.gameObject.GetComponent<EnvironmentWorldGeneration>();
                Debug.Log(environmentWorldGeneration);
                EditableWorld world = environmentWorldGeneration.world;

                int blockType = VoxelWorld.AIR;
                if (invSolid > 0) {
                    blockType = VoxelWorld.SOLID;
                    invSolid -= 1;
                } else if (invPrecious > 0) {
                    blockType = VoxelWorld.SOLID_PRECIOUS;
                    invPrecious -= 1;
                }
                world.SetBlock(roundedX, roundedY, roundedZ, blockType);
            }
        }

        private void removeBlock(){
            var result = executeRay();
            if (result != null){
                (Vector3 pos, Vector3 lastPos) = result.Value;

                int roundedX = Mathf.FloorToInt(pos.x);
                int roundedY = Mathf.FloorToInt(pos.y);
                int roundedZ = Mathf.FloorToInt(pos.z);
                
                EnvironmentWorldGeneration environmentWorldGeneration = transform.parent.gameObject.GetComponent<EnvironmentWorldGeneration>();
                Debug.Log("remove:");
                Debug.Log(environmentWorldGeneration);
                EditableWorld world = environmentWorldGeneration.world;

                int blockType = world.BlockAt(roundedX, roundedY, roundedZ);
                
                if (blockType == VoxelWorld.SOLID_PRECIOUS) {
                    invPrecious += 1;
                } else {
                    invSolid += 1;
                }
                Debug.Log("precious: "+invPrecious + " ; solid: " + invSolid);
                world.SetBlock(roundedX, roundedY, roundedZ, VoxelWorld.AIR);
            }
        }

        private (Vector3, Vector3)? executeRay(){
            float step = checkIncrement;
            Vector3 lastPos = new Vector3();
            EnvironmentWorldGeneration environmentWorldGeneration = transform.parent.gameObject.GetComponent<EnvironmentWorldGeneration>();
            Debug.Log("ray:");
            Debug.Log(environmentWorldGeneration);
            EditableWorld world = environmentWorldGeneration.world;

            while (step < range) {
                Vector3 pos = playerCamera.transform.position + (playerCamera.transform.forward * step);
                Debug.Log(pos);
                pos -= environmentWorldGeneration.transform.position;
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
}