using System.Collections;
using System.Collections.Generic;
using Voxels;
using UnityEngine;

[CreateAssetMenu(fileName = "NewPerlinWorldGenerator.Asset", menuName = "WorldGenerator/Perlin")]
public class PerlinWorld : VoxelWorld
{
    public PerlinWorld(int xMax, int yMax, int zMax) : base(xMax, yMax, zMax)
    {}

    public override int BlockAt(int x, int y, int z)
    {
        bool outOfBounds = x < 0 || y < 0 || z < 0 || x >= xMax || y >= yMax || z >= zMax;
        if (outOfBounds)
        {
            return VoxelWorld.AIR;
        }

        float perlinValue = Mathf.PerlinNoise(seed * 100 + x * 0.1f, seed * 100 + z * 0.1f); // Generate Perlin noise
        if (y < 10 + perlinValue * 10)
        {
            return VoxelWorld.SOLID;
        }

        return VoxelWorld.AIR;
    }
}
