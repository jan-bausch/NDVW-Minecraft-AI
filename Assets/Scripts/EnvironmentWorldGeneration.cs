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
            VoxelRenderer.RenderWorld(gameObject, world, material);
        }
    }
}
