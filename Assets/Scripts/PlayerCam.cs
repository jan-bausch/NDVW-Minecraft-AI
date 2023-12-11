using UnityEngine;

namespace Player
{
    public class PlayerCam : MonoBehaviour
    {
        public float sensX;
        public float sensY;

        float xRotation;
        float yRotation;

        public Transform orientation;

        void Start()
        {
            // Lock and hide cursor
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        void Update()
        {
            float mouseX = Input.GetAxisRaw("Mouse X") * Time.deltaTime * sensX;
            float mouseY = Input.GetAxisRaw("Mouse Y") * Time.deltaTime * sensY;

            yRotation += mouseX;
            xRotation -= mouseY;

            xRotation = Mathf.Clamp(xRotation, -90f, 90f);

            transform.rotation = Quaternion.Euler(xRotation, yRotation, 0);
            if (orientation)
            {
                orientation.rotation = Quaternion.Euler(0, yRotation, 0);
            }
        }
    }
}