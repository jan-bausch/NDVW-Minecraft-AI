using UnityEngine;
using WorldEditing;
using Voxels;

namespace Environment {
    public class EnvironmentWorldGeneration : MonoBehaviour, EditableWorldListener
    {
        public enum WorldTypes
        {
            Flat,
            Perlin
        }

        public Material material;

        public WorldTypes worldType = WorldTypes.Flat;
        public int worldSizeX = 20;
        public int worldSizeY = 40;
        public int worldSizeZ = 20;

        public EditableWorld world;

        void Generate(int seed)
        {
            worldSizeX = PlayerPrefs.GetInt("SizeOfWorld", 20);
            worldSizeY = PlayerPrefs.GetInt("SizeOfWorld", 40);
            worldSizeZ = PlayerPrefs.GetInt("SizeOfWorld", 20);

            VoxelWorld baseWorld = null;
            if (worldType == WorldTypes.Flat)
            {
                baseWorld = new FlatWorld(worldSizeX, worldSizeY, worldSizeZ);
            } else if (worldType == WorldTypes.Perlin)
            {
                baseWorld = new PerlinWorld(worldSizeX, worldSizeY, worldSizeZ);
            }

            world = new EditableWorld(baseWorld);
            world.OverrideSeed(seed);
            world.Subscribe(this);
            VoxelRenderer.RenderWorld(gameObject, world, material);

            Transform playerTransform = transform.Find("Player");
            // playerTransform.position = transform.position + randomSpawnPos(seed, 5, 10, 5, 10);
            // Spawn player in the upper part of the bottom left quadrant
            playerTransform.position = transform.position + randomSpawnPos(seed, 
                                                                           Mathf.FloorToInt(worldSizeX / 8), 
                                                                           Mathf.FloorToInt(3 * worldSizeX / 8), 
                                                                           Mathf.FloorToInt(2 * worldSizeZ / 8), 
                                                                           Mathf.FloorToInt(4 * worldSizeZ / 8));

            for (int i = 1; i <= PlayerPrefs.GetInt("NumberOfCreepers"); i++)
            {
                Transform creeperTransform = transform.Find("Creeper"+i);
                Vector3 creeperPos = Vector3.zero;
                // Spawn creeper in upper right quadrant 
                if (i == 1) {
                    creeperPos = transform.position + randomSpawnPos(seed, 
                                                                     Mathf.FloorToInt(5 *worldSizeX / 8), 
                                                                     Mathf.FloorToInt(7 * worldSizeX / 8), 
                                                                     Mathf.FloorToInt(5 * worldSizeZ / 8), 
                                                                     Mathf.FloorToInt(7 * worldSizeZ / 8));
                }
                // Spawn creeper in upper left quadrant 
                else if (i == 2) {
                    creeperPos = transform.position + randomSpawnPos(seed, 
                                                                     Mathf.FloorToInt(worldSizeX / 8), 
                                                                     Mathf.FloorToInt(3 * worldSizeX / 8), 
                                                                     Mathf.FloorToInt(5 * worldSizeZ / 8), 
                                                                     Mathf.FloorToInt(7 * worldSizeZ / 8));
                }
                // Spawn creeper in lower right quadrant 
                else if (i == 3) {
                    creeperPos = transform.position + randomSpawnPos(seed, 
                                                                     Mathf.FloorToInt(5 *worldSizeX / 8), 
                                                                     Mathf.FloorToInt(7 * worldSizeX / 8), 
                                                                     Mathf.FloorToInt(worldSizeZ / 8), 
                                                                     Mathf.FloorToInt(3 * worldSizeZ / 8));
                }
                // Spawn creeper in the lower part of the bottom left quadrant
                else if (i == 4) {
                    creeperPos = transform.position + randomSpawnPos(seed, 
                                                                     Mathf.FloorToInt(worldSizeX / 8), 
                                                                     Mathf.FloorToInt(3 * worldSizeX / 8), 
                                                                     Mathf.FloorToInt(0 * worldSizeZ / 8), 
                                                                     Mathf.FloorToInt(2 * worldSizeZ / 8));
                }
                creeperTransform.position = creeperPos;
            }
            Physics.SyncTransforms();
        }

        private Vector3 randomSpawnPos(int seed, int startX, int endX, int startZ, int endZ)
        {

            worldSizeX = PlayerPrefs.GetInt("SizeOfWorld", 20);
            worldSizeY = PlayerPrefs.GetInt("SizeOfWorld", 40);
            worldSizeZ = PlayerPrefs.GetInt("SizeOfWorld", 20);
            
            System.Random random = new System.Random(seed);

            int spawnX = random.Next(startX, endX);
            int spawnZ = random.Next(startZ, endZ);
            int spawnY = 0;
            int prevprev = VoxelWorld.AIR;
            int prev = VoxelWorld.AIR;
            for (; spawnY < worldSizeY+1; spawnY++)
            {
                int block = world.BlockAt(spawnX, spawnY, spawnZ);
                if (block == VoxelWorld.AIR && prev == VoxelWorld.AIR && prevprev != VoxelWorld.AIR) break;
                prevprev = prev;
                prev = block;
            }

            return new Vector3(spawnX + 0.5f, spawnY + 1.0f, spawnZ + 0.5f);
        }

        public void OnBlockUpdate(int x, int y, int z, int oldBlock, int newBlock)
        {
            //Debug.Log("hey");
            //Debug.Log(this);

            // For some obscure reason, this is way faster
            VoxelRenderer.RenderWorld(gameObject, world, material);
            
            //VoxelRenderer.UpdateVoxel(gameObject, world, x, y, z);
        }
    }
}