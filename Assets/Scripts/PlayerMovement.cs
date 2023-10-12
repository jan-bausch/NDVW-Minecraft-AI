using System.Collections;
using System.Collections.Generic;
using UnityEngine;


[System.Serializable]
public class Boundary
{
    public float xMin, xMax, yMin, yMax, zMin, zMax;
}
public class PlayerMovement : MonoBehaviour
{
    private Vector3 CameraPosition;
    [Header("Camera Settings")]
    public float CameraSpeed;
    public Boundary boundary;
    // Start is called before the first frame update
    void Start()
    {
        CameraPosition = this.transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow)){
            CameraPosition.z += CameraSpeed / 100;
        } 
        if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow)){
            CameraPosition.z -= CameraSpeed / 100;
        }
        if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow)){
            CameraPosition.x += CameraSpeed / 100;
        }
        if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow)){
            CameraPosition.x -= CameraSpeed / 100;
        }
        if (Input.GetKey(KeyCode.Space)){
            CameraPosition.y += CameraSpeed / 100;
        } else {
            CameraPosition.y -= CameraSpeed / 100;
        }
        CameraPosition.x = Mathf.Clamp(CameraPosition.x, boundary.xMin, boundary.xMax);
        CameraPosition.y = Mathf.Clamp(CameraPosition.y, boundary.yMin, boundary.yMax);
        CameraPosition.z = Mathf.Clamp(CameraPosition.z, boundary.zMin, boundary.zMax);
        this.transform.position = CameraPosition;
    }
}
