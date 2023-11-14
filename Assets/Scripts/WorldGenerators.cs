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
}

