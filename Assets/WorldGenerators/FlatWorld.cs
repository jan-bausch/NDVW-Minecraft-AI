using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Voxels
{
    public class FlatWorld : VoxelWorld
    {
        public int threshold;

        public FlatWorld(int xMax, int yMax, int zMax) : base(xMax, yMax, zMax)
        {
            this.threshold = yMax/2;
        }

        public override int BlockAt(int x, int y, int z)
        {
            bool outOfBounds = x < 0 || y < 0 || z < 0 || x >= xMax || y >= yMax || z >= zMax;
            if (outOfBounds)
            {
                return VoxelWorld.AIR;
            }
            
            Random.InitState(seed + x * 10000 + y * 100 + z);
            
            if ((Random.value < 0.05 && y <= threshold) || (Random.value < 0.02 && y > threshold+4))
            {
                return VoxelWorld.SOLID_PRECIOUS;
            }
            if (y > threshold)
            {
                return VoxelWorld.AIR;
            }
            return VoxelWorld.SOLID;
        }
    }
}