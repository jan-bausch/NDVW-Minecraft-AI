using UnityEngine;
using WorldGenerators;
using WorldEditing;
using Voxels;

namespace Environment {
    public class EnvironmentWorldGeneration : MonoBehaviour
    {
        public Material material;
        public VoxelWorld baseWorld;
        public EditableWorld world;

        void Generate(int seed)
        {
            world = new EditableWorld(baseWorld);
            Debug.Log(gameObject);
            Debug.Log(world);
            Debug.Log(material);
            world.OverrideSeed(seed);
            VoxelRenderer.RenderWorld(gameObject, world, material);
        }

    }
}