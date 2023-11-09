using UnityEngine;
using System.Collections;

namespace WorldFunctions {
    public abstract class WorldGenerator
    {
        public const int BLOCK_TYPES = 2;
        public const int AIR = 0;
        public const int SOLID = 1;
        public const int SOLID_PRECIOUS = 2;

        public int xMax, yMax, zMax;

        public WorldGenerator(int xMax, int yMax, int zMax)
        {
            this.xMax = xMax;
            this.yMax = yMax;
            this.zMax = zMax;
        }

        public abstract int Generate(int x, int y, int z);
    }

    public class FlatWorld : WorldGenerator
    {
        private int threshold;

        public FlatWorld(int xMax, int yMax, int zMax, int threshold) : base(xMax, yMax, zMax)
        {
            this.threshold = threshold;
        }

        public override int Generate(int x, int y, int z)
        {
            bool outOfBounds = x < 0 || y < 0 || z < 0 || x >= xMax || y >= yMax || z >= zMax;
            if (y > threshold || outOfBounds)
            {
                return WorldGenerator.AIR;
            }
            if (x % 10 == 0 && y == threshold && z % 10 == 0)
            {
                return WorldGenerator.SOLID_PRECIOUS;
            }
            return WorldGenerator.SOLID;
        }
    }

    public class PerlinWorld : WorldGenerator
    {
        public PerlinWorld(int xMax, int yMax, int zMax) : base(xMax, yMax, zMax){}

        public override int Generate(int x, int y, int z)
        {
            bool outOfBounds = x < 0 || y < 0 || z < 0 || x >= xMax || y >= yMax || z >= zMax;
            if (outOfBounds)
            {
                return WorldGenerator.AIR;
            }

            float perlinValue = Mathf.PerlinNoise(x * 0.1f, z * 0.1f); // Generate Perlin noise
            if (y < 10 + perlinValue * 10)
            {
                return WorldGenerator.SOLID;
            }

            return WorldGenerator.AIR;
        }
    }
}