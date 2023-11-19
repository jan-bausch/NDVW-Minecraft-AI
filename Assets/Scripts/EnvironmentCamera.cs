using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Environment {
    public class EnvironmentCamera : MonoBehaviour
    {
        // Parameters for camera setup
        public float fieldOfView = 60f;
        public float nearClipPlane = 0.3f;
        public float farClipPlane = 1000f;
        public Vector3 relativePosition = new Vector3(10f, 25f, 10f);
        public Vector3 rotationEulerAngles = new Vector3(90f, 0f, 0f); // Rotation in euler angles

        private RenderTexture renderTexture;

        void Start()
        {
            // Create a new camera
            GameObject newCamera = new GameObject("DownwardCamera");
            Camera cameraComponent = newCamera.AddComponent<Camera>();

            // Set camera parameters
            cameraComponent.fieldOfView = fieldOfView;
            cameraComponent.nearClipPlane = nearClipPlane;
            cameraComponent.farClipPlane = farClipPlane;

            // Set the position relative to the GameObject this script is attached to
            newCamera.transform.SetParent(transform); // Attaches the camera to the same parent as the GameObject
            newCamera.transform.localPosition = relativePosition; // Position relative to the parent

            // Set the rotation
            newCamera.transform.localRotation = Quaternion.Euler(rotationEulerAngles);

            // Set the camera to render downwards
            // newCamera.transform.LookAt(transform.position - Vector3.up, Vector3.up);

            // Create a new render texture
            renderTexture = new RenderTexture(64, 64, 24);
            renderTexture.name = "DownwardRenderTexture";

            // Set the camera to render into the render texture
            cameraComponent.targetTexture = renderTexture;

            // Assign the render texture to a material or use it as needed
            // Example: Assigning it to a UI RawImage's texture
            // rawImage.texture = renderTexture;
        }

        public int[] GetPixelsGrayscale()
        {
            if (renderTexture == null) return null;
        
            RenderTexture.active = renderTexture;

            // Create a texture2D and read pixels
            Texture2D tex = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            tex.Apply();

            Color[] pixels = tex.GetPixels();
            int[] averagedRGB = new int[pixels.Length];

            for (int i = 0; i < pixels.Length; i++)
            {
                // Calculate the average of R, G, B components and pack them into a single int
                averagedRGB[i] = ((int)(pixels[i].r * 255f) + (int)(pixels[i].g * 255f) + (int)(pixels[i].b * 255f)) / 3;
                //averagedRGB[i] = (int)(pixels[i].r * 255f);
            }

            RenderTexture.active = null;
            
            return averagedRGB;
        }
    }
}
