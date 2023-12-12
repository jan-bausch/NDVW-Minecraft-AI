using UnityEngine;

namespace Player
{
    public class PlayerCam : MonoBehaviour
    {
        public float sensX;
        public float sensY;

        float xRotation;
        float yRotation;

        public bool remoteControlled = false;

        public bool lookingLeft = false;
        public bool lookingRight = false;
        public bool lookingUp = false;
        public bool lookingDown = false;

        public Transform orientation;

        void Start()
        {
            // Lock and hide cursor
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        void Update()
        {
            if (!remoteControlled)
            {
                moveUpdate(Time.deltaTime, Input.GetAxisRaw("Mouse X"), Input.GetAxisRaw("Mouse Y"));
            }
        }

        public void MoveUpdate(float delta)
        {
            float axisX = 0.0f;
            float axisY = 0.0f;
            if (lookingLeft) axisX = -1.0f;
            if (lookingRight) axisX = 1.0f;
            if (lookingDown) axisY = -1.0f;
            if (lookingUp) axisY = 1.0f;

            moveUpdate(delta, axisX, axisY);
        }

        private void moveUpdate(float delta, float axisX, float axisY)
        {
            float mouseX = axisX * delta * sensX;
            float mouseY = axisY * delta * sensY;

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