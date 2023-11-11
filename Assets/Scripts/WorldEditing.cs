using System.Collections;
using System;
using System.Collections.Generic;
using UnityEngine;
using Voxels;

namespace WorldEditing
{
    public class EditableWorld : VoxelWorld
    {
        private Dictionary<Vector3, int> editedBlocks;
        private VoxelWorld world;

        public EditableWorld(VoxelWorld world) : base(
            world.xMax, 
            world.yMax, 
            world.zMax
        )
        {   
            this.editedBlocks = new Dictionary<Vector3, int>();
            this.world = world;
        }

        public override int BlockAt(int x, int y, int z)
        {
            int block = 0;
            Vector3 key = new Vector3(x, y, z);
            if (editedBlocks.TryGetValue(key, out block))
            {
                return block;
            }
            return world.BlockAt(x, y, z);
        }

        public void SetBlock(int x, int y, int z, int block)
        {
            Vector3 key = new Vector3(x, y, z);
            editedBlocks[key] = block;
        }
    }
}
