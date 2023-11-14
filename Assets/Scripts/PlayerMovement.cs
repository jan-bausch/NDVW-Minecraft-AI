using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
    public float moveSpeed = 5.0f;
    public float jumpForce = 5.0f;
    private CharacterController controller;
    private Vector3 moveDirection;

    void Start()
    {
        controller = GetComponent<CharacterController>();
    }

    void Update()
    {
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");

        if (controller.isGrounded)
        {
            if (Input.GetButton("Jump"))
            {
                moveDirection.y = jumpForce;
            }
        }

        moveDirection.x = horizontal * moveSpeed;
        moveDirection.y -= 9.81f * Time.deltaTime;
        moveDirection.z = vertical * moveSpeed;

        controller.Move(moveDirection * Time.deltaTime);
    }
}