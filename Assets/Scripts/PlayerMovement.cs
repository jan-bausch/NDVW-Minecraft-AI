using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Player {
public class PlayerMovement : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed = 7.0f;
    public float jumpForce = 5.0f;
    public float airMultiplier = 0.5f;
    public float gravity = 9.81f;

    [Header("Controls")]
    public bool remoteControlled = false;
    public bool jumping = false;
    public bool goingLeft = false;
    public bool goingRight = false;
    public bool goingForward = false;
    public bool goingBackward = false;
    private float horizontalInput = 0.0f;
    private float verticalInput = 0.0f;

    public bool dead = false;
    
    [Header("Orientation")]
    public Transform orientation;

    private CharacterController controller;
    private Vector3 moveDirection;

    void Start()
    {
        controller = GetComponent<CharacterController>();
    }

    void Update()
    {
        if (!remoteControlled)
        {
            MoveUpdate(Time.deltaTime);
        }
    }

    public void MoveUpdate(float delta){
        if (remoteControlled && !dead)
        {
            verticalInput = 0.0f;
            horizontalInput = 0.0f;
            if (goingBackward) verticalInput = -1.0f;
            if (goingForward) verticalInput = 1.0f;
            if (goingLeft) horizontalInput = -1.0f;
            if (goingRight) horizontalInput = 1.0f;
        } else
        {
            jumping = Input.GetKey(KeyCode.Space);
            horizontalInput = Input.GetAxisRaw("Horizontal");
            verticalInput = Input.GetAxisRaw("Vertical");
        }

        // Calculate x and z in function of the orientation 
        Vector3 tempVector = orientation.forward * verticalInput * moveSpeed + orientation.right * horizontalInput * moveSpeed;
        moveDirection.x = tempVector.x;
        moveDirection.z = tempVector.z;
        
        // Check if player is grounded
        if (controller.isGrounded) {
            if (jumping) {
                moveDirection.y = jumpForce;
            }
            moveDirection.y -= gravity * delta;
            controller.Move(moveDirection * delta);
        } else {
            moveDirection.y -= gravity * delta;
            controller.Move(moveDirection * delta * airMultiplier);
        }
    }
}

}