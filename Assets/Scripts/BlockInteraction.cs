using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlockInteraction : MonoBehaviour
{
    [Header("Camera")]
    public Camera playerCamera;
    public float rayTime = 1.0f;
    public float range = 5.0f;

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0)) // Assuming left mouse button for breaking
        {
            Ray ray = playerCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            Debug.DrawRay(ray.origin, ray.direction * 100f, Color.red, rayTime);

            if (Physics.Raycast(ray, out hit, range))
            {
                Vector3 hitPointOnMesh = hit.point;

                // Round the coordinates tFloorToInto integers
                int roundedX = Mathf.(hitPointOnMesh.x);
                int roundedY = Mathf.FloorToInt(hitPointOnMesh.y);
                int roundedZ = Mathf.FloorToInt(hitPointOnMesh.z);

                Vector3 roundedHitPoint = new Vector3(roundedX, roundedY, roundedZ);

                Debug.Log(roundedHitPoint);
            }
        }

    }
}
