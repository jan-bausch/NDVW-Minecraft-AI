using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Voxels;

[CreateAssetMenu(fileName = "NewFlatWorldGenerator.Asset", menuName = "WorldGenerator/Flat")]
public class FlatWorld : VoxelWorld
{
    public int threshold;

    public FlatWorld(int xMax, int yMax, int zMax, int threshold) : base(xMax, yMax, zMax)
    {
        this.threshold = threshold;
    }

    public override int BlockAt(int x, int y, int z)
    {
        bool outOfBounds = x < 0 || y < 0 || z < 0 || x >= xMax || y >= yMax || z >= zMax;
        if (y > threshold || outOfBounds)
        {
            return VoxelWorld.AIR;
        }
        Random.InitState(seed + x * 10000 + y * 100 + z);
        
        if (Random.value < 0.3 && y <= threshold+4)
        {
            return VoxelWorld.SOLID_PRECIOUS;
        }
        return VoxelWorld.SOLID;
    }
}
