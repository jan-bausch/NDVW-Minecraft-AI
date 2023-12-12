using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Voxels
{
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

            float perlinValue = Mathf.PerlinNoise(seed * 100 + x * 0.2f, seed * 100 + z * 0.2f); 
            float normalizedY = (float)y / 25.0f;

            if (y < 10 + perlinValue * 10 * normalizedY)
            {
                float preciousProbability = Mathf.Lerp(0.001f, 0.02f, normalizedY);
                Random.InitState(seed + x * 10000 + y * 100 + z);

                if (Random.value < preciousProbability)
                {
                    return VoxelWorld.SOLID_PRECIOUS;
                }
                else
                {
                    return VoxelWorld.SOLID;
                }
            }

            return VoxelWorld.AIR;
        }
    }
}