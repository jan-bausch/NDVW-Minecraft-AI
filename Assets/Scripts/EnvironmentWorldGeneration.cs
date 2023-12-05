using UnityEngine;
using WorldGenerators;
using WorldEditing;
using Voxels;

namespace Environment {
    public class EnvironmentWorldGeneration : MonoBehaviour, EditableWorldListener
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
            world.Subscribe(this);
            VoxelRenderer.RenderWorld(gameObject, world, material);
        }

        public GameObject getGameObject(){
            return gameObject;
        }

        public void OnBlockUpdate(int x, int y, int z, int oldBlock, int newBlock)
        {
            Debug.Log("WORLD UPDATE");
            VoxelRenderer.RenderWorld(gameObject, world, material);
        }
    }
}