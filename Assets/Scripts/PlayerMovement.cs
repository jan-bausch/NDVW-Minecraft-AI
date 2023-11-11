using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
    public float moveSpeed = 5.0f; // Adjust this to set the movement speed

    void Update()
    {
        //Debug.Log((int)(1.0f / Time.smoothDeltaTime));
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        float elevation = 0;

        if (Input.GetKey(KeyCode.Space))
        {
            elevation = 1;
        }
        else if (Input.GetKey(KeyCode.LeftShift))
        {
            elevation = -1;
        }

        Vector3 movement = new Vector3(horizontal, elevation, vertical) * moveSpeed * Time.deltaTime;
        transform.Translate(movement, Space.Self);
    }
}
