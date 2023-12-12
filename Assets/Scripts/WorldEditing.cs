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

        private int _seed;

        private List<EditableWorldListener> listeners;

        public EditableWorld(VoxelWorld world) : base(
            world.xMax, 
            world.yMax, 
            world.zMax
        )
        {   
            this.editedBlocks = new Dictionary<Vector3, int>();
            this.world = world;
            listeners = new List<EditableWorldListener>();
        }

        public override void OverrideSeed(int seed)
        {
            this._seed = seed;
            world.OverrideSeed(seed);
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
            Debug.Log("[World "+_seed+"]: Block set ("+ x + "," + y + "," + z +") " + block);
            int oldBlock = BlockAt(x, y, z);
            Vector3 key = new Vector3(x, y, z);
            editedBlocks[key] = block;

            foreach(var listener in listeners)
            {
                Debug.Log(listener);
                listener.OnBlockUpdate(x, y, z, oldBlock, block);
            }
        }

        public void Subscribe(EditableWorldListener listener)
        {
            Debug.Log("[World "+_seed+"]: Listener added ("+ (listeners.Count+1) +")");
            listeners.Add(listener);
        }

        public int GetSeed()
        {
            return _seed;
        }
    }

    public interface EditableWorldListener 
    {
        void OnBlockUpdate(int x, int y, int z, int oldBlock, int newBlock);
    }
}
