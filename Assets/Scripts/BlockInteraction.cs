using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlockInteraction : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0)) // Assuming left mouse button for breaking
        {
            Ray ray = new Ray(transform.position, Vector3.down);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, Mathf.Infinity))
            {
                Vector3 hitPointOnMesh = hit.point;

                // Round the coordinates to integers
                int roundedX = Mathf.RoundToInt(hitPointOnMesh.x);
                int roundedY = Mathf.RoundToInt(hitPointOnMesh.y);
                int roundedZ = Mathf.RoundToInt(hitPointOnMesh.z);

                Vector3 roundedHitPoint = new Vector3(roundedX, roundedY, roundedZ);

                Debug.Log(roundedHitPoint);
            }
        }

    }
}
