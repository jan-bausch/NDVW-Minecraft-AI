using UnityEngine;
using WorldGenerators;
using WorldEditing;
using Voxels;

public class WorldGeneration : MonoBehaviour
{
    public int worldCount;
    public Material material;
    public VoxelWorld baseWorld;
    private EditableWorld world;

    void Start()
    {
        world = new EditableWorld(baseWorld);
        VoxelRenderer.RenderWorld(gameObject, world, material);
    }
}
