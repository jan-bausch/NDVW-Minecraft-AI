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
        public int worldSizeY = 20;
        public int worldSizeZ = 20;

        public EditableWorld world;

        void Generate(int seed)
        {
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
            playerTransform.position = transform.position + randomSpawnPos(seed, 5, 10, 5, 10);

            Transform creeperTransform = transform.Find("Creeper");
            Vector3 creeperPos = transform.position + randomSpawnPos(seed, worldSizeX-5, worldSizeX-1, worldSizeZ-5, worldSizeZ-1);
            creeperTransform.position = creeperPos;
            Physics.SyncTransforms();
        }

        private Vector3 randomSpawnPos(int seed, int startX, int endX, int startZ, int endZ)
        {
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
            VoxelRenderer.RenderWorld(gameObject, world, material);
        }
    }
}