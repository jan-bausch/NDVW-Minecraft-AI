using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WorldFunctions;

public class WorldGeneration : MonoBehaviour
{
    public int xMax, yMax, zMax, threshold;
    public Material material;

    // Start is called before the first frame update
    void Start()
    {
        // FlatWorld generator = new FlatWorld(xMax, yMax, zMax, threshold);
        PerlinWorld generator = new PerlinWorld(xMax, yMax, zMax);

        for (int x = 0; x < xMax; x++){
            for (int y = 0; y < yMax; y++){
                for (int z = 0; z < zMax; z++){
                    CreateBlock(generator,x,y,z);
                }
            }
        }

        MeshFilter[] meshFilters = GetComponentsInChildren<MeshFilter>();
        CombineInstance[] combine = new CombineInstance[meshFilters.Length];

        int i = 0;
        while (i < meshFilters.Length)
        {
            combine[i].mesh = meshFilters[i].sharedMesh;
            combine[i].transform = meshFilters[i].transform.localToWorldMatrix;
            meshFilters[i].gameObject.SetActive(false);
            i++;
        }

        Mesh mesh = new Mesh();
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.CombineMeshes(combine);

        // Vector2[] uvs = new Vector2[mesh.vertices.Length];

        // float uvScale = 4f;

        // for (i = 0; i < mesh.vertices.Length; i++) {
        //     uvs[i] = new Vector2(
        //         mesh.vertices[i].x * uvScale,
        //         mesh.vertices[i].y * uvScale
        //     );
        // }

        // mesh.uv = uvs;

        MeshFilter meshFilter = gameObject.AddComponent<MeshFilter>();
        meshFilter.sharedMesh = mesh;
        MeshRenderer meshRenderer = gameObject.AddComponent<MeshRenderer>();
        meshRenderer.material = material;

        MeshCollider meshCollider = gameObject.AddComponent<MeshCollider>();
        meshCollider.sharedMesh = mesh;
        // transform.gameObject.SetActive(true);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void CreateBlock(WorldGenerator generator, int x, int y, int z){
        int blockType = generator.Generate(x,y,z);
        if (blockType == WorldGenerator.AIR) return;

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
            20, 22, 21,
            20, 23, 22
        };

        for (int i = 0; i < vertices.Length; i++)
        {
            vertices[i] = vertices[i] + new Vector3(x, y, z); 
            uvs[i].y = (uvs[i].y / (float) WorldGenerator.BLOCK_TYPES) 
                + (blockType * (1.0f / (float) WorldGenerator.BLOCK_TYPES));
        }

        bool removeFrontFace = generator.Generate(x - 0, y - 0, z - 1) != WorldGenerator.AIR;
        bool removeBackFace = generator.Generate(x - 0, y - 0, z + 1) != WorldGenerator.AIR;
        bool removeTopFace = generator.Generate(x - 0, y + 1, z - 0) != WorldGenerator.AIR;
        bool removeBottomFace = generator.Generate(x - 0, y - 1, z - 0) != WorldGenerator.AIR;
        bool removeLeftFace = generator.Generate(x - 1, y - 0, z - 0) != WorldGenerator.AIR;
        bool removeRightFace = generator.Generate(x + 1, y - 0, z - 0) != WorldGenerator.AIR;

        List<int> filteredTriangles = new List<int>();

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
                filteredTriangles.Add(triangles[i]);
                filteredTriangles.Add(triangles[i + 1]);
                filteredTriangles.Add(triangles[i + 2]);
                filteredTriangles.Add(triangles[i + 3]);
                filteredTriangles.Add(triangles[i + 4]);
                filteredTriangles.Add(triangles[i + 5]);
            }
        }

        int[] renderedFaces = filteredTriangles.ToArray();

        if (renderedFaces.Length == 0) return;
        // Create a new mesh
        Mesh mesh = new Mesh();

        // Assign vertices and triangles to the mesh
        mesh.vertices = vertices;
        mesh.triangles = renderedFaces;
        mesh.SetUVs(0, uvs);

        // Create a new game object and add a MeshFilter and MeshRenderer component
        GameObject obj = new GameObject("CustomMesh");
        obj.transform.SetParent(transform);
        MeshFilter meshFilter = obj.AddComponent<MeshFilter>();

        // Assign the created mesh to the MeshFilter
        meshFilter.mesh = mesh;
    }
}