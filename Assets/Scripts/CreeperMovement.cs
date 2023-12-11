using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Creeper
    {
    public class CreeperMovement : MonoBehaviour
    {
        [Header("Target")]
        public Transform target;
        public float detectionRadius = 8.0f;

        [Header("Movement")]
        public bool remoteControlled;
        public float moveSpeed = 7.0f;
        public float jumpForce = 8.0f;
        public float airMultiplier = 0.5f;
        public float gravity = 9.81f;

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

        public void MoveUpdate(float delta)
        {
            if (target && ((target.position - transform.position).magnitude < detectionRadius))
            {
                // Target player
                transform.LookAt(new Vector3(target.position.x, transform.position.y, target.position.z));

                // Move forward
                Vector3 tempVector = transform.forward * moveSpeed;
                moveDirection.x = tempVector.x;
                moveDirection.z = tempVector.z;
            } else {
                moveDirection.x -= 0.1f;
                moveDirection.z -= 0.1f;
                moveDirection.x = Mathf.Clamp(moveDirection.x, 0, 10.0f);
                moveDirection.z = Mathf.Clamp(moveDirection.z, 0, 10.0f);
            }

            // Check if creeper is grounded
            if (controller.isGrounded) {
                if (Random.Range(0.0f, 10.0f) < 0.1f) {
                    //Debug.Log("Jump");
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