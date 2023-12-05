using UnityEngine;
using WorldGenerators;
using WorldEditing;
using Voxels;

namespace Environment {
    public class EnvironmentWorldGeneration : MonoBehaviour
    {
        public Material material;
        public VoxelWorld baseWorld;
        private EditableWorld world;

        void Generate(int seed)
        {
            world = new EditableWorld(baseWorld);
            world.OverrideSeed(seed);
            var random = new System.Random(seed);
            VoxelRenderer.RenderWorld(gameObject, world, material);
            Transform playerTransform = transform.Find("Player");
            int spawnX = random.Next(5, 16);
            int spawnZ = random.Next(5, 16);
            int spawnY = 0;
            int prevprev = VoxelWorld.AIR;
            int prev = VoxelWorld.AIR;
            for (; spawnY < 21; spawnY++)
            {
                int block = world.BlockAt(spawnX, spawnY, spawnZ);
                if (block == VoxelWorld.AIR && prev == VoxelWorld.AIR && prevprev != VoxelWorld.AIR) break;
                prevprev = prev;
                prev = block;
            }
            playerTransform.position = transform.position + new Vector3(spawnX + 0.5f, spawnY + 0.5f, spawnZ + 0.5f);
        }
    }
}
