using System;
using UnityEngine;
using System.Collections.Generic;
using Voxels;

namespace WorldGenerators 
{
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
            if (x % 10 == 0 && y == threshold && z % 10 == 0)
            {
                return VoxelWorld.SOLID_PRECIOUS;
            }
            return VoxelWorld.SOLID;
        }
    }
}

