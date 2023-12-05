using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed = 7.0f;
    public float jumpForce = 5.0f;
    public float airMultiplier = 0.5f;
    public float gravity = 9.81f;

    [Header("Keybinds")]
    public KeyCode jumpKey = KeyCode.Space;
    
    [Header("Orientation")]
    public Transform orientation;

    private float horizontalInput;
    private float verticalInput;

    private CharacterController controller;
    private Vector3 moveDirection;

    void Start()
    {
        controller = GetComponent<CharacterController>();
    }

    void Update()
    {
        HandleInput();
        MovePlayer();
    }

    private void HandleInput(){
        horizontalInput = Input.GetAxisRaw("Horizontal");
        verticalInput = Input.GetAxisRaw("Vertical");
    }

    private void MovePlayer(){
        // Calculate x and z in function of the orientation 
        Vector3 tempVector = orientation.forward * verticalInput * moveSpeed + orientation.right * horizontalInput * moveSpeed;
        moveDirection.x = tempVector.x;
        moveDirection.z = tempVector.z;

        // Check if player is grounded
        if (controller.isGrounded) {
            if (Input.GetKey(jumpKey)) {
                moveDirection.y = jumpForce;
            }
            moveDirection.y -= gravity * Time.deltaTime;
            controller.Move(moveDirection * Time.deltaTime);
        } else {
            moveDirection.y -= gravity * Time.deltaTime;
            controller.Move(moveDirection * Time.deltaTime * airMultiplier);
        }
    }
}