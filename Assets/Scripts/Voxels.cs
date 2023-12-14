using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Voxels
{
    public abstract class VoxelWorld : ScriptableObject
    {
        public const int BLOCK_TYPES = 2;
        public const int AIR = -1;
        public const int SOLID = 0;
        public const int SOLID_PRECIOUS = 1;

        public int xMax, yMax, zMax;

        protected int seed;

        List<Vector3> verticesList = new List<Vector3> ();
        List<int> trianglesList = new List<int> ();
        List<Vector2> uvsList = new List<Vector2> ();

        public VoxelWorld(int xMax, int yMax, int zMax)
        {
            this.xMax = xMax;
            this.yMax = yMax;
            this.zMax = zMax;

            var random = new System.Random();
            this.seed = random.Next();
        }

        public virtual void OverrideSeed(int seed)
        {
            this.seed = seed;
        }

        public abstract int BlockAt(int x, int y, int z);
    }

    public class VoxelRenderer
    {
        public static void RenderWorld(GameObject parent, VoxelWorld world, Material material)
        { 
            MeshFilter meshFilter = parent.GetComponent<MeshFilter>();
            if (meshFilter != null) UnityEngine.Object.Destroy(meshFilter.mesh);
            
            MeshRenderer meshRenderer = parent.GetComponent<MeshRenderer>();
            if (meshRenderer != null) UnityEngine.Object.Destroy(meshRenderer.material);

            MeshCollider meshCollider = parent.GetComponent<MeshCollider>();
            if (meshCollider != null) UnityEngine.Object.Destroy(meshCollider.sharedMesh);
            
            List<Vector3> vertices = new List<Vector3>();
            List<int> renderedFaces = new List<int>();
            List<Vector2> uvs = new List<Vector2>();
            List<Vector3> normals = new List<Vector3>();
            for (int x = 0; x < world.xMax; x++){
                for (int y = 0; y < world.yMax; y++){
                    for (int z = 0; z < world.zMax; z++){
                        //int vertexCount = vertices.Count;
                        //Debug.Log(vertexCount);
                        int voxelId = x * world.yMax * world.zMax + y * world.zMax + z;
                        var (voxelVertices, voxelRenderedFaces, voxelUvs, voxelNormals) = renderVoxel(world,x,y,z);
                        //Debug.Log(voxelVertices.Length);
                        
                        for (int i = 0; i < voxelVertices.Length; i++)
                        {
                            vertices.Add(voxelVertices[i]);
                        }
                        for (int i = 0; i < voxelRenderedFaces.Length; i++)
                        {
                            //Debug.Log(vertexCount);
                            renderedFaces.Add(voxelRenderedFaces[i] + voxelId * 24);
                        }
                        for (int i = 0; i < voxelUvs.Length; i++)
                        {
                            uvs.Add(voxelUvs[i]);
                        }
                        for (int i = 0; i < voxelNormals.Length; i++)
                        {
                            normals.Add(voxelNormals[i]); 
                        }
                    }
                }
            }

            Mesh mesh = new Mesh();

            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.vertices = vertices.ToArray();
            mesh.triangles = renderedFaces.ToArray();
            mesh.normals = normals.ToArray();
            mesh.SetUVs(0, uvs.ToArray());
 
            if (meshFilter == null)
            {
                meshFilter = parent.AddComponent<MeshFilter>();
            }
            meshFilter.sharedMesh = mesh;

            if (meshRenderer == null)
            {
                meshRenderer = parent.AddComponent<MeshRenderer>();
            }
            meshRenderer.material = material;

            if (meshCollider == null)
            {
                meshCollider = parent.AddComponent<MeshCollider>();
            }
            meshCollider.sharedMesh = mesh;
        }

        public static void UpdateVoxel(GameObject parent, VoxelWorld world, int x, int y, int z)
        {
            MeshFilter meshFilter = parent.GetComponent<MeshFilter>();
            Mesh mesh = meshFilter.sharedMesh;
            int voxelId = x * world.yMax * world.zMax + y * world.zMax + z;

            var originalTriangles = mesh.triangles.ToList();
            Func<int, bool> filter = vertexId => vertexId < voxelId*24 && vertexId >= (voxelId+1)*24;
            var triangles = originalTriangles.Where(filter).ToList();

            var (voxelVertices, voxelRenderedFaces, voxelUvs, voxelNormals) = renderVoxel(world,x,y,z);
            
            for (int i = 0; i < voxelRenderedFaces.Length; i++)
            {
                triangles.Add(voxelRenderedFaces[i] + voxelId * 24);
            }

            meshFilter.sharedMesh = mesh;
            MeshCollider meshCollider = parent.GetComponent<MeshCollider>();
            meshCollider.sharedMesh = mesh;
        }

        private static (Vector3[], int[], Vector2[], Vector3[]) renderVoxel(VoxelWorld world, int x, int y, int z)
        {
            int blockType = world.BlockAt(x,y,z);

            Vector3[] vertices = new Vector3[]
            {
                new Vector3(0, 0, 0), // 0
                new Vector3(1, 0, 0), // 1
                new Vector3(1, 1, 0), // 2
                new Vector3(0, 1, 0), // 3
                new Vector3(1, 0, 0), // 4
                new Vector3(1, 0, 1), // 5
                new Vector3(1, 1, 1), // 6
                new Vector3(1, 1, 0), // 7
                new Vector3(1, 0, 1), // 8
                new Vector3(0, 0, 1), // 9
                new Vector3(0, 1, 1), // 10
                new Vector3(1, 1, 1), // 11
                new Vector3(0, 0, 1), // 12
                new Vector3(0, 0, 0), // 13
                new Vector3(0, 1, 0), // 14
                new Vector3(0, 1, 1), // 15
                new Vector3(1, 1, 1), // 16
                new Vector3(0, 1, 1), // 17
                new Vector3(0, 1, 0), // 18
                new Vector3(1, 1, 0), // 19
                new Vector3(1, 0, 0), // 20
                new Vector3(0, 0, 0), // 21
                new Vector3(0, 0, 1), // 22
                new Vector3(1, 0, 1), // 23
            };

            Vector2[] uvs = new Vector2[]
            {
                new Vector2(0, 0.5f), // 0
                new Vector2(0.25f, 0.5f), // 1
                new Vector2(0.25f, 1), // 2
                new Vector2(0, 1), // 3
                new Vector2(0.25f, 0.5f), // 4
                new Vector2(0.5f, 0.5f), // 5
                new Vector2(0.5f, 1), // 6
                new Vector2(0.25f, 1), // 7
                new Vector2(0.5f, 0.5f), // 8
                new Vector2(0.75f, 0.5f), // 9
                new Vector2(0.75f, 1), // 10
                new Vector2(0.5f, 1), // 11
                new Vector2(0.75f, 0.5f), // 12
                new Vector2(1, 0.5f), // 13
                new Vector2(1, 1), // 14
                new Vector2(0.75f, 1), // 15
                new Vector2(0.25f, 0), // 16
                new Vector2(0.5f, 0), // 17
                new Vector2(0.5f, 0.5f), // 18
                new Vector2(0.25f, 0.5f), // 19
                new Vector2(0, 0), // 20
                new Vector2(0.25f, 0), // 21
                new Vector2(0.25f, 0.5f), // 22
                new Vector2(0, 0.5f), // 23
            };

            // Define triangles for the cube's six faces
            int[] triangles = new int[]
            {
                // Front face
                0, 2, 1,
                0, 3, 2,

                // Back face
                8, 10, 9,
                8, 11, 10,

                // Top face
                16, 18, 17,
                16, 19, 18,

                // Bottom face
                20, 22, 21,
                20, 23, 22,

                // Left face
                12, 14, 13,
                12, 15, 14,

                // Right face
                4, 6, 5,
                4, 7, 6 
            };

            Vector3[] normalsPerFace = new Vector3[]
            {
                // Front
                new Vector3(0, 0, -1),  
                // Back
                new Vector3(0, 0, 1),
                // Top
                new Vector3(0, 1, 0), 
                // Bottom
                new Vector3(0, -1, 0), 
                // Left
                new Vector3(-1, 0, 0), 
                // Right
                new Vector3(1, 0, 0) 
            };

            List<Vector3> normalsList = new List<Vector3>();
            for (int i = 0; i < vertices.Length; i++)
            {
                vertices[i] = vertices[i] + new Vector3(x, y, z); 
                uvs[i].y = ((uvs[i].y * 64.0f) + ((float) (blockType % 4) * 80.0f)) / 304.0f;
                uvs[i].x = ((uvs[i].x * 128.0f) + ((float) (blockType / 4) * 144.0f)) / 272.0f;
                normalsList.Add(new Vector3(0, 0, 0));
            }

            for (int i = 0; i < triangles.Length; i += 6)
            {
                for (int j = 0; j < 6; j++)
                { 
                    int vertexId = triangles[i+j];
                    Vector3 faceNormal = normalsPerFace[(i/6)];
                    normalsList[vertexId] = faceNormal;
                }
            }

            Vector3[] normals = normalsList.ToArray();

            if (blockType == VoxelWorld.AIR)
            {
                return (vertices, new int[]{}, uvs, normals);
            }

            bool removeFrontFace = world.BlockAt(x - 0, y - 0, z - 1) != VoxelWorld.AIR;
            bool removeBackFace = world.BlockAt(x - 0, y - 0, z + 1) != VoxelWorld.AIR;
            bool removeTopFace = world.BlockAt(x - 0, y + 1, z - 0) != VoxelWorld.AIR;
            bool removeBottomFace = world.BlockAt(x - 0, y - 1, z - 0) != VoxelWorld.AIR;
            bool removeLeftFace = world.BlockAt(x - 1, y - 0, z - 0) != VoxelWorld.AIR;
            bool removeRightFace = world.BlockAt(x + 1, y - 0, z - 0) != VoxelWorld.AIR;

            List<int> filteredTriangles = new List<int>();
            List<Vector3> filteredVerticesNormals = new List<Vector3>();

            for (int i = 0; i < vertices.Length; i++)
            {
                filteredVerticesNormals.Add(new Vector3(0, 0, 0));
            }

            for (int i = 0; i < triangles.Length; i += 6)
            {
                int faceCheck = i / 6;
                if ((faceCheck == 0 && !removeFrontFace) ||
                    (faceCheck == 1 && !removeBackFace) ||
                    (faceCheck == 2 && !removeTopFace) ||
                    (faceCheck == 3 && !removeBottomFace) ||
                    (faceCheck == 4 && !removeLeftFace) ||
                    (faceCheck == 5 && !removeRightFace))
                {
                    for (int j = 0; j < 6; j++)
                    {   
                        int vertexId = triangles[i+j];
                        filteredTriangles.Add(vertexId);
                    }   
                }
            }

            int[] renderedFaces = filteredTriangles.ToArray();

            return (vertices, renderedFaces, uvs, normals);

            // if (renderedFaces.Length == 0) return null;
            // // Create a new mesh
            // Mesh mesh = new Mesh();

            // // Assign vertices and triangles to the mesh
            // mesh.vertices = vertices;
            // mesh.triangles = renderedFaces;
            // mesh.normals = normals;
            // mesh.SetUVs(0, uvs);

            // // Create a new game object and add a MeshFilter and MeshRenderer component
            // GameObject obj = new GameObject("Voxel_" + x + "_" + y + "_" + "_" + z);
            // MeshFilter meshFilter = obj.AddComponent<MeshFilter>();

            // // Assign the created mesh to the MeshFilter
            // meshFilter.mesh = mesh;

            // return obj;
        }
   }
}