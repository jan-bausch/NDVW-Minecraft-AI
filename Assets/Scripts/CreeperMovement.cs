using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CreeperMovement : MonoBehaviour
{

    public Transform target;
    public float moveSpeed = 2.5f;
    public float jumpForce = 5.0f;
    public float detectionRadius = 5.0f;

    private CharacterController controller;

    void Start()
    {
        controller = GetComponent<CharacterController>();
    }

    void Update()
    {
        Vector3 moveDirection = new Vector3();

        moveDirection.y -= 9.81f * Time.deltaTime;

        if (controller.isGrounded)
        {
            if (Input.GetButton("Jump"))
            {
                moveDirection.y = jumpForce * Time.deltaTime;
            }
        }

        // Check if player is close
        if (target && ((target.transform.position - transform.position).magnitude < detectionRadius))
        {
            // Target player
            transform.LookAt(new Vector3(target.position.x, transform.position.y, target.position.z));

            // Move forward
            moveDirection += transform.forward * Time.deltaTime * moveSpeed;
        }

        controller.Move(moveDirection);
    }
}
